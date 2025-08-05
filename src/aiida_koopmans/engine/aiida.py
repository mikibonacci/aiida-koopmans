import fnmatch
import logging
import os
import pathlib
from pathlib import Path
import tempfile
import time
from typing import Any, ClassVar, Generator, Literal, overload

from aiida_pseudo.data.pseudo import UpfData
from ase_koopmans.spectrum.band_structure import BandStructure
import dill as pickle
from koopmans.calculators import (
    Calc,
    KoopmansCPCalculator,
    ProjwfcCalculator,
    PW2WannierCalculator,
    Wannier90Calculator,
)
from koopmans.engines.engine import Engine
from koopmans.engines.localhost import LocalhostEngine
from koopmans.files import File, LocalFile
from koopmans.process_io import IOModel
from koopmans.processes import CommandLineTool, Process, ProcessProtocol
from koopmans.processes.wjl import (
    WannierJLCheckNeighborsProcess,
    WannierJLGenerateNeighborsProcess,
    WannierJLSplitBlockOutput,
    WannierJLSplitProcess,
)
from koopmans.pseudopotentials import read_pseudo_file
from koopmans.status import Status
import numpy as np
from pydantic import Field

from aiida import load_profile, orm
from aiida.engine import submit

from aiida_koopmans.engine.step_data import AiiDAStepData, AiiDAStepsData
from aiida_koopmans.utils import get_builder_from_ase, read_output_file


class AiiDAEngine(Engine):

    """
    ProcessProtocol data is a dictionary containing the following information:
    step_data = {calc.directory: {'workchain': workchain, 'remote_folder': remote_folder}}
    and any other info we need for AiiDA.
    """

    name: ClassVar[str] = "AiiDAEngine"
    blocking: bool = True
    step_data: AiiDAStepsData = Field(default_factory=lambda: AiiDAStepsData())

    def model_post_init(self, context: Any, /) -> None:
        load_profile()

    def run(self, step: ProcessProtocol, additional_flags: list[str] = []):
        logger = logging.getLogger(__name__)

        if isinstance(
            step, Process
        ):  # Process are not AiiDA calculations (more like merging files and so on)
            if isinstance(step, CommandLineTool):
                # prepare shell job
                code, arguments, outputs = prepare_shell_job(
                    process=step, step_data=self.step_data
                )
                # launch shell job.
                from aiida_shell import launch_shell_job

                results, node = launch_shell_job(
                    code,
                    arguments=arguments,
                    # outputs=outputs,
                    submit=True,
                )
                logger.info(
                    f"aiida-koopmans running shelljob {node.pk} for step {step.uid}"
                )
                self.set_status(step, Status.RUNNING)
                self.step_data.steps[step.uid].workchain = node.pk
                return
            elif isinstance(
                step,
                (
                    WannierJLGenerateNeighborsProcess,
                    WannierJLSplitProcess,
                    WannierJLCheckNeighborsProcess,
                ),
            ):

                # Run the step in a second temporary directory
                with tempfile.TemporaryDirectory() as t:
                    # Copy over any input files to a local temporary directory so Julia can access them
                    tmp_directory = pathlib.Path(t)
                    for key, inp in step.inputs:
                        if not isinstance(inp, File):
                            continue

                        # When reading, read from the remote directory and use the 'aiida' prefix
                        content: str | bytes
                        try:
                            content = self.read_file(
                                inp.with_stem("aiida"), binary=False
                            )
                        except UnicodeDecodeError:
                            content = self.read_file(
                                inp.with_stem("aiida"), binary=True
                            )

                        # When writing, save to the temporary directory and use the original prefix
                        local_file = LocalFile(
                            tmp_directory / "remote_files" / ("wannier90" + inp.suffix)
                        )
                        self.write_local_file(content, local_file, parents=True)

                        # Override the input with the local file
                        setattr(step.inputs, key, local_file)

                    orig_directory, step.directory = step.directory, tmp_directory

                    # Because the step is identified by its directory, we need to also add
                    # this temporary directory to step_data
                    self.step_data.steps[step.uid] = self.step_data.steps[
                        str(orig_directory)
                    ]

                    # Temporarily override step.engine (so that engine-dependent functionality
                    # within step.run (e.g. file operations) are done locally
                    local_engine = LocalhostEngine()
                    step.engine = local_engine
                    try:
                        step.run()
                    except:
                        os.system("cp -r " + str(tmp_directory) + " ~/tmp/aiida-tmp")
                        raise ValueError()

                    # Reset the engine
                    step.engine = self

                    def store_file(file: File) -> None:
                        """Store a file.

                        Do this by reading the local files from within the tmp directory
                        # and then writing them back (which stores them as SinglefileData objects in AiiDA)
                        """

                        # Read from the tmp directory and use the original prefix
                        step.directory = tmp_directory
                        content = self.read_local_file(file, binary=False)

                        # Save to the original directory and use the 'aiida' prefix
                        step.directory = orig_directory
                        filename = (
                            str(file.name)
                            .replace("/", "-")
                            .replace("wannier90", "aiida")
                        )
                        stored_file = File(step, filename)
                        self.write_file(content, stored_file)
                        step.directory = tmp_directory

                        return stored_file

                    def store_outputs(obj: Any) -> Any:
                        if isinstance(obj, list):
                            out = []
                            for item in obj:
                                stored_item = store_outputs(item)
                                out.append(stored_item)
                            return out
                        elif isinstance(obj, Process):
                            obj.outputs = store_outputs(obj.outputs)
                            return obj
                        elif isinstance(obj, IOModel):
                            for field in obj.model_fields:
                                stored_file = store_outputs(getattr(obj, field))
                                setattr(obj, field, stored_file)
                            return obj
                        elif isinstance(obj, File):
                            return store_file(obj)
                        else:
                            return obj

                    store_outputs(step)
                    # for key, output in step.outputs:

                    #     if isinstance(output, list):
                    #         for item in output:

                    #     if isinstance(output, WannierJLSplitBlockOutput):
                    #         raise ValueError()
                    #         orig_prefix, tmp_prefix = block.win_file.prefix, 'aiida'
                    #         for file in block:

                    #         # Restore the original prefix
                    #         block.prefix = orig_prefix
                    #     elif isinstance(output, File):
                    #         # When reading, read from the tmp directory and use the original prefix
                    #         step.directory = tmp_directory
                    #         content = self.read_local_file(output, binary=False)

                    #         # When writing, save to the original directory and use the 'aiida' prefix
                    #         step.directory = orig_directory
                    #         self.write_file(content, output.with_stem('aiida'))

                # Copy the step_data back to the original directory
                self.step_data.steps[str(orig_directory)] = self.step_data.steps.pop(
                    step.uid
                )

                # Reset the step directory
                step.directory = orig_directory

                # Re-dump outputs (now using the correct engine and directory)
                step.dump_outputs()

                self.set_status(step, Status.COMPLETED)
                return

            else:
                step.run()
                self.set_status(step, Status.COMPLETED)
                return

        if step.prefix in ["wannier90_preproc", "pw2wannier90"]:
            self.set_status(step, Status.COMPLETED)
            return

        builder, self.step_data = get_builder_from_ase(
            calculator=step, step_data=self.step_data
        )  # ASE to AiiDA conversion. put some error message if the conversion fails
        running = submit(builder)
        logger.info(
            f"aiida-koopmans submitted workchain {running.pk} for step {step.uid}"
        )
        # running = aiidawrapperwchain.submit(builder) # in the non-blocking case.

        # The below will be passed to the context, so we will need to store also the instance of the submitted workchain, if in KoopmansWorkChain.
        self.step_data.steps[step.uid].workchain = running.pk

        self.set_status(step, Status.RUNNING)
        self._step_running_message(step, end="\n")

        return

    def load_step_data(self):
        # TODO: if all steps in the step data are completed, we do not need to run this method.
        try:
            with open("step_data.pkl", "rb") as f:
                # this will overwrite the step_data[configuration],
                # i.e. if we change codes or res we will not see it if
                # the file already exists.
                step_data = pickle.load(f)
        except FileNotFoundError:
            return

        # here we update the configuration if it is provided in the engine_config file.
        # useful if we want to restart with different resources.
        # step_data['configuration'] = self.step_data.pop('configuration', step_data['configuration'])
        step_data["configuration"].update(self.step_data.configuration)

        for step_uid in step_data["steps"]:
            if step_uid not in self.step_data.steps:
                self._step_skipped_message_by_uid(step_uid)

        self.step_data = AiiDAStepsData(**step_data)

    def dump_step_data(self):
        step_data = self.step_data.model_dump()
        with open("step_data.pkl", "wb") as f:
            pickle.dump(step_data, f)

    def _get_status(self, step: ProcessProtocol) -> Status:
        status = self.get_status_by_uid(step.uid)
        return status

    def get_status_by_uid(self, uid: str) -> Status:
        self.load_step_data()
        if uid not in self.step_data.steps:
            self.step_data.steps[uid] = AiiDAStepData(status=Status.NOT_STARTED)
        status = self.step_data.steps[uid].status
        logger = logging.getLogger(__name__)
        logger.debug(f"Fetching status of step with UID {uid}; it is {status}")
        return status

    def _set_status(self, step: ProcessProtocol, status: Status):
        self.set_status_by_uid(step.uid, status)

    def set_status_by_uid(self, uid: str, status: Status):
        self.step_data.steps[uid].status = status
        self.dump_step_data()
        logger = logging.getLogger(__name__)
        logger.debug(f"Setting status of step with UID {uid} to {status}")

    def _update_statuses(self) -> None:

        for uid in self.step_data.steps:

            if self.get_status_by_uid(uid) == Status.FAILED:
                raise ValueError(f"The step {uid} failed.")
            elif not self.get_status_by_uid(uid) == Status.RUNNING:
                continue

            workchain = orm.load_node(self.step_data.steps[uid].workchain)
            if workchain.is_finished_ok:
                if self.get_status_by_uid(uid) != Status.COMPLETED:
                    self._step_completed_message_by_uid(uid)
                    self.set_status_by_uid(uid, Status.COMPLETED)

            elif workchain.is_finished or workchain.is_excepted or workchain.is_killed:
                self.set_status_by_uid(uid, Status.FAILED)
                self._step_failed_message_by_uid(uid)
                raise ValueError(f"Workchain {workchain.pk} failed.")

            return

    def _steps_are_running(self) -> bool:
        """Check if any step is running."""
        self.update_statuses()
        return Status.RUNNING in [v.status for v in self.step_data.steps.values()]

    def load_results(self, step: ProcessProtocol) -> None:

        # TODO: if the step is completed, we do not need to run this load_results method.
        self.load_step_data()

        if isinstance(step, Process):
            if isinstance(step, CommandLineTool):
                if orm.load_node(
                    self.step_data.steps[step.uid].workchain
                ).is_finished_ok:
                    step._set_outputs()
                    return
            try:
                # Check if step.outputs has been defined
                step.outputs
            except ValueError:
                step.load_outputs()
            return

        if step.prefix in ["wannier90_preproc", "pw2wannier90"]:
            self.set_status(step, Status.COMPLETED)
            return
        workchain = orm.load_node(self.step_data.steps[step.uid].workchain)
        if "remote_folder" in workchain.outputs:
            self.step_data.steps[
                step.uid
            ].remote_folder = workchain.outputs.remote_folder.pk
        output = None
        if isinstance(step, Wannier90Calculator):
            if hasattr(workchain.outputs, "wannier90"):
                output_node = workchain.outputs.wannier90
            else:
                output_node = workchain.outputs
            output = read_output_file(step, output_node.retrieved)
            self.step_data.steps[step.uid].remote_folder = output_node.remote_folder.pk
        elif step.ext_out in [".pwo", ".w2ko", ".kso", ".kho", ".cpo"]:
            output_node = workchain.outputs
            output = read_output_file(step, output_node.retrieved)
            if hasattr(output.calc, "kpts"):
                step.kpts = output.calc.kpts
        else:
            output_node = workchain.outputs
            output = read_output_file(step, output_node.retrieved)

        if step.ext_out in [
            ".pwo",
            ".pro",
            ".wout",
            ".w2ko",
            ".kso",
            ".kho",
            ".cpo",
            ".wko",
        ]:
            step.calc = output.calc
            step.results = output.calc.results
            # if step.ext_out == ".pwo": step.generate_band_structure() #nelec=int(workchain.outputs.output_parameters.get_dict()['number_of_electrons']))

        if "interpolated_bands" in output_node:
            bs = output_node.interpolated_bands
            path = step.parameters["kpoint_path"]
            bands = bs.get_bands()
            if len(bands.shape) == 2:
                bands = np.expand_dims(bands, axis=0)
            step.results["band structure"] = BandStructure(path=path, energies=bands)

        # This maybe is not the right place to do this, but we need to set the results of the step.
        if step.ext_out in [".cpo"]:
            step.results["lambda"] = step.read_ham_files()
            if step.parameters.do_bare_eigs:
                step.results["bare lambda"] = step.read_ham_files(bare=True)

        step._post_run()
        self.dump_step_data()

        # self._step_completed_message(step)

    def load_old_calculator(self, calc: Calc):
        raise NotImplementedError  # load_old_calculator(calc)

    def get_pseudopotential(self, library: str, element: str):

        qb = orm.QueryBuilder()
        qb.append(orm.Group, filters={"label": {"==": library}}, tag="pseudo_group")
        qb.append(
            UpfData,
            filters={"attributes.element": {"==": element}},
            with_group="pseudo_group",
        )

        pseudo_data = None

        for pseudo in qb.all():
            with tempfile.TemporaryDirectory() as dirpath:
                temp_file = pathlib.Path(dirpath) / (
                    pseudo[0].base.attributes.all["element"] + ".upf"
                )
                with pseudo[0].open(pseudo[0].filename, "rb") as handle:
                    temp_file.write_bytes(handle.read())

                pseudo_data = read_pseudo_file(temp_file)

        if not pseudo_data:
            raise ValueError(
                f"Could not find pseudopotential for element {element} in library {library}"
            )

        pseudo_data.filename = library + "/" + pseudo_data.filename.name

        return pseudo_data

    def link_pseudopotential(self, step, src: Path, dest: Path) -> None:
        """Link a pseudopotential file to the provided step.

        For AiiDA, the linking of pseudopotentials is done later by setting builder.pseudos.
        We will set builder.pseudos to self.step_data.pseudo_family"""

        family = str(src.parent)

        if self.step_data.pseudo_family:
            if family != self.step_data.pseudo_family:
                raise ValueError(
                    f"Cannot link pseudopotential {src} to step {step.uid} because it belongs to a different family ({family} != {self.step_data.pseudo_family})"
                )
        else:
            self.step_data.pseudo_family = family

    def read_ham_file(
        self, calculator: KoopmansCPCalculator, filename: Path
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:

        new_filename = Path(str(filename).split("/")[-1])

        from koopmans.calculators._koopmans_cp import read_ham_file

        hamiltonian = None
        retrieved = orm.load_node(
            self.step_data.steps[calculator.uid]["workchain"]
        ).outputs.retrieved

        with tempfile.TemporaryDirectory() as dirpath:
            for _filename in retrieved.base.repository.list_object_names():
                if str(new_filename) == _filename:
                    # Create the file with the desired name
                    output_file = pathlib.Path(dirpath) / _filename
                    with retrieved.open(_filename, "rb") as handle:
                        output_file.write_bytes(handle.read())

                    hamiltonian = read_ham_file(output_file)

        return hamiltonian

    def read_file(self, file: File, binary: bool = False) -> str | bytes:
        """Read the content of a file from the AiiDA database."""

        # Patch reading from pw2wannier90 calculations, as when running with AiiDA these
        # calculations are replaced by a workchain

        if isinstance(file.parent_process, PW2WannierCalculator):
            [wannier90_uid] = [
                uid
                for uid, step in self.step_data.steps.items()
                if uid.startswith(str(file.parent_process.directory.parent))
                and step.workchain is not None
            ]
            step_data = self.step_data.steps[wannier90_uid]
        else:
            step_data = self.step_data.steps[file.parent_process.uid]

        mode = "rb" if binary else "r"
        if step_data.workchain is None:
            if str(file.name) not in step_data.input_files:
                raise FileNotFoundError(f"File {file.name} does not exist.")
            singlefiledata = orm.load_node(
                identifier=step_data.input_files[str(file.name)]
            )
            return singlefiledata.get_content(mode=mode)

        workchain = orm.load_node(step_data.workchain)
        filename = (
            str(file.name).replace(file.parent_process.prefix, "aiida").split("/")[-1]
        )

        # additional replace for kc.kcw_hr_occ/emp.dat
        if "kcw_hr" in filename:
            filename = filename.replace(
                file.parent_process.parameters.prefix + ".", "aiida."
            )
        elif "wannier90_hr" in filename:
            filename = filename.replace("wannier90_hr", "aiida_hr")

        if "wannier90" in file.parent_process.prefix:
            if hasattr(workchain.outputs, "wannier90"):
                wannier90_node = workchain.outputs.wannier90
            else:
                wannier90_node = workchain.outputs
            content = wannier90_node.retrieved.get_object_content(filename, mode=mode)
        else:
            mode = (
                "rb" if ((".dat" in filename or ".xml" in filename) and binary) else "r"
            )
            content = workchain.outputs.retrieved.get_object_content(
                filename, mode=mode
            )
        return content

    @overload
    def read_local_file(self, file: File, binary: Literal[True]) -> bytes:
        ...

    @overload
    def read_local_file(self, file: File, binary: Literal[False]) -> str:
        ...

    @overload
    def read_local_file(self, file: File, binary: bool = False) -> str | bytes:
        ...

    def read_local_file(self, file: File, binary: bool = False) -> str | bytes:
        """Read the content of a file from the local filesystem."""
        assert file.parent_process.absolute_directory is not None
        full_path: Path = file.parent_process.absolute_directory / file.name
        fstring = "rb" if binary else "r"
        with open(full_path, fstring) as f:
            content = f.read()
        return content

    def write_local_file(
        self, content: str | bytes, file: File, parents: bool = False
    ) -> None:
        """Write the content to a file in the local filesystem."""
        assert file.parent_process.absolute_directory is not None
        full_path: Path = file.parent_process.absolute_directory / file.name
        fstring = "wb" if isinstance(content, bytes) else "w"

        if parents:
            full_path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.info(
            f'Writing {"bytes" if isinstance(content, bytes) else "string"} to {full_path}'
        )
        with open(full_path, fstring) as f:
            f.write(content)

    def write_file(self, content: str | bytes, file: File) -> None:
        if "inputs.pkl" in str(file.name):
            return

        filename = str(file.name).replace("/", "-")

        if isinstance(content, bytes):
            try:
                singlefile = orm.SinglefileData.from_bytes(content, filename)
            except AttributeError as e:
                import io

                singlefile = orm.SinglefileData(io.BytesIO(content), filename)
        else:
            singlefile = orm.SinglefileData.from_string(content, filename)
        singlefile.store()
        self.step_data.steps[file.parent_process.uid].input_files[
            str(file.name)
        ] = singlefile.pk
        return singlefile

    def glob(
        self, directory: File, pattern: str, recursive: bool = False
    ) -> Generator[File, None, None]:

        workchain = orm.load_node(
            self.step_data.steps[directory.parent_process.uid].workchain
        )
        if "wannier90" in getattr(directory.parent_process, "prefix", ""):
            listnames = (
                workchain.outputs.wannier90.retrieved.base.repository.list_object_names()
            )
        else:
            listnames = workchain.outputs.retrieved.base.repository.list_object_names()

        for name in listnames:
            tomatch = str(directory.name / pattern)
            if hasattr(directory.parent_process, "prefix"):
                tomatch = tomatch.replace(directory.parent_process.prefix, "aiida")
            if isinstance(
                directory.parent_process, ProjwfcCalculator
            ):  # TODO: this is a hack, we need to find a better way to do this.
                tomatch = tomatch.replace(
                    directory.parent_process.parameters.filpdos, "aiida"
                )
            if fnmatch.fnmatch(name, tomatch):
                yield File(directory.parent_process, pathlib.Path(name))

    def available_pseudo_libraries(self) -> set[str]:
        """Return a set of available pseudopotential libraries."""
        raise NotImplementedError("Not implemented yet")

    def install_pseudopotential(self, file: Path, library: str) -> None:
        """Install a local file so that it can be accessed by the engine via self.get_pseudopotential()."""
        raise NotImplementedError("Not implemented yet")

    def uninstall_pseudopotential_library(self, library: str) -> None:
        """Uninstall a pseudopotential library."""
        raise NotImplementedError("Not implemented yet")

    def chdir(self, directory: Path):
        """Return a context manager that changes directory (returning to the original directory when finished)."""
        raise NotImplementedError("Not needed for AiiDA engine")

    def mkdir(
        self, directory: File, parents: bool = False, exist_ok: bool = False
    ) -> None:
        """Create a directory; should mimic Path.mkdir."""
        raise NotImplementedError("Not needed for AiiDA engine")

    def rmdir(self, directory: File) -> None:
        """Remove a directory; should mimic Path.rmdir."""
        raise NotImplementedError("Not needed for AiiDA engine")

    def copy_file(
        self, source: File, destination: File, exist_ok: bool = False
    ) -> None:
        """Copy a file; should mimic shutil.copy."""
        raise NotImplementedError("Not needed for AiiDA engine")

    def link_file(
        self,
        source: File,
        destination: File,
        recursive: bool = False,
        overwrite: bool = False,
    ) -> None:
        """Create a symbolic link at destination that points to source; should mimic Path.symlink_to.

        Additionally must support recursive = True, which, if source is a directory, will link all the files
        individually rather than creating a single link to the entire directory"
        """
        raise NotImplementedError("Not needed for AiiDA engine")

    def unlink_file(self, file: File) -> None:
        """Remove a file; should mimic Path.unlink."""
        raise NotImplementedError("Not needed for AiiDA engine")

    def file_exists(self, file: File) -> bool:
        """Check if a file exists; should mimic Path.exists."""
        return True
        raise NotImplementedError("Not needed for AiiDA engine")

    def file_is_dir(self, file: File) -> bool:
        """Check if a file is a directory; should mimic Path.is_dir."""
        raise NotImplementedError("Not needed for AiiDA engine")

    def rename_file(self, src: File, dst: File) -> None:
        """Rename a file; should mimic Path.rename."""
        raise NotImplementedError("Not needed for AiiDA engine")

    def wait(self, seconds: float) -> None:
        """Wait for a specified amount of time, dumping directories while waiting."""
        start_time = time.time()
        self._dump_all_steps(timeout=seconds)
        remaining_time = seconds - (time.time() - start_time)
        if remaining_time > 0:
            time.sleep(remaining_time)

    def _dump_all_steps(self, timeout: float | None = None) -> None:
        logger = logging.getLogger(__name__)
        start_time = time.time()
        for uid, step in self.step_data.steps.items():
            current_time = time.time()
            if timeout is not None and current_time - start_time > timeout:
                break
            if (
                step.status == Status.COMPLETED
                and not Path(uid).exists()
                and step.workchain is not None
            ):
                logger.info(f"Dumping data for {uid}")
                proc = orm.load_node(step.workchain)
                proc.dump(output_path=uid, include_outputs=True)

        return

    def _teardown(self) -> None:
        """Perform any necessary engine-specific cleanup actions when the workflow is completed."""
        self._dump_all_steps()
