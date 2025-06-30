from koopmans.engines.engine import Engine
from koopmans.processes import ProcessProtocol
from koopmans.calculators import Calc, ProjwfcCalculator, KoopmansCPCalculator
from koopmans.pseudopotentials import read_pseudo_file
from koopmans.status import Status
from koopmans.files import File
from koopmans.processes import Process, CommandLineTool

from typing import Generator, List, Any
from pathlib import Path
from pydantic import BaseModel, Field
from typing import ClassVar

from aiida.engine import run_get_node, submit

from aiida_koopmans.utils import *

from aiida_pseudo.data.pseudo import UpfData

import time

import dill as pickle
import pathlib
import tempfile
import fnmatch

from aiida import orm, load_profile
load_profile()

class AiiDAStepData(BaseModel):
    """
    This class is used to store the step data in a dictionary.
    It contains the following information:
    - step_data = {calc.directory: {'workchain': workchain, 'remote_folder': remote_folder}}
    and any other info we need for AiiDA.
    """
    configuration: dict
    steps: dict[str, dict] = Field(default_factory=dict)
    pseudo_family: str | None = None
    structure: int | None = None
    
    #def some model validator or computed field:
        # here we add the logic to populate configuration by default
        # 1. we look for codes stored in AiiDA at localhost, e.g. pw-version@localhost,
        # 2. we look for codes in the PATH,
        # 3. if we don't find the code in AiiDA db but in the PATH, we store it in AiiDA db.
        # 4. if we don't find the code in AiiDA db and in the PATH and not configuration is provided, we raise an error.
        # 5. if no resource info in configuration, we try to look at PARA_PREFIX env var.


class AiiDAEngine(Engine):

    """
    ProcessProtocol data is a dictionary containing the following information:
    step_data = {calc.directory: {'workchain': workchain, 'remote_folder': remote_folder}}
    and any other info we need for AiiDA.
    """
    name: ClassVar[str] = "AiiDAEngine"
    blocking: bool = True
    step_data: AiiDAStepData
    
    def run(self, step: ProcessProtocol):

        self.get_status(step)

        if isinstance(step, Process): # Process are not AiiDA calculations (more like merging files and so on)
            if isinstance(step, CommandLineTool):
                # prepare shell job
                code, arguments, outputs = prepare_shell_job(process=step, step_data=self.step_data)
                # launch shell job.
                from aiida_shell import launch_shell_job
                results, node = launch_shell_job(
                    code,
                    arguments=arguments,
                    #outputs=outputs,
                    submit=True,
                )
                print(f"Running shelljob {node.pk} for step {step.uid}")
                self.step_data.steps[step.uid] = {'workchain': node.pk, }
                self.set_status(step, Status.RUNNING)
                return
            else:
                step.run()
                self.set_status(step, Status.COMPLETED)
                self._step_completed_message(step)
                return
            
        if step.prefix in ['wannier90_preproc', 'pw2wannier90']:
            self.set_status(step, Status.COMPLETED)
            return
        
        self.step_data.steps[step.uid] = {} # maybe not needed

        builder, self.step_data = get_builder_from_ase(calculator=step, step_data=self.step_data) # ASE to AiiDA conversion. put some error message if the conversion fails
        running = submit(builder)
        print(f"Running workchain {running.pk} for step {step.uid}")
        # running = aiidawrapperwchain.submit(builder) # in the non-blocking case.
        
        # The below will be passed to the context, so we will need to store also the instance of the submitted workchain, if in KoopmansWorkChain.
        self.step_data.steps[step.uid] = {'workchain': running.pk, } #'remote_folder': running.outputs.remote_folder}

        self.set_status(step, Status.RUNNING)

        return

    def load_step_data(self):
        # TODO: if all steps in the step data are completed, we do not need to run this method.
        try:
            with open('step_data.pkl', 'rb') as f:
                # this will overwrite the step_data[configuration],
                # i.e. if we change codes or res we will not see it if
                # the file already exists.
                step_data = pickle.load(f)
                # here we update the configuration if it is provided in the engine_config file.
                # useful if we want to restart with different resources.
                #step_data['configuration'] = self.step_data.pop('configuration', step_data['configuration'])
                step_data['configuration'].update(self.step_data.configuration)
                
                self.step_data = AiiDAStepData(**step_data)
        except FileNotFoundError:
            pass

    def dump_step_data(self):
        step_data = self.step_data.model_dump()
        with open('step_data.pkl', 'wb') as f:
            pickle.dump(step_data, f)

    def get_status(self, step: ProcessProtocol) -> Status:
        status = self.get_status_by_uid(step.uid)
        #print(f"Getting status for step {step.uid}: {status}")
        return status
        
    
    def get_status_by_uid(self, uid: str) -> Status:
        
        self.load_step_data()
        if uid not in self.step_data.steps:
            self.step_data.steps[uid] = {'status': Status.NOT_STARTED}
        return self.step_data.steps[uid]['status']

    def set_status(self, step: ProcessProtocol, status: Status):
        self.set_status_by_uid(step.uid, status)
        
    def set_status_by_uid(self, uid: str, status: Status):
        self.step_data.steps[uid]['status'] = status
        self.dump_step_data()

    def update_statuses(self) -> None:
        
        for uid in self.step_data.steps:

            if self.get_status_by_uid(uid) == Status.FAILED:
                raise ValueError(f"The step {uid} failed.")
            elif not self.get_status_by_uid(uid) == Status.RUNNING:
                continue
            else:
                time.sleep(5)
            
            workchain = orm.load_node(self.step_data.steps[uid]['workchain'])
            if workchain.is_finished_ok:
                self._step_completed_message_by_uid(uid)
                self.set_status_by_uid(uid, Status.COMPLETED)

            elif workchain.is_finished or workchain.is_excepted or workchain.is_killed:
                self._step_failed_message_by_uid(uid)
                self.set_status_by_uid(uid, Status.FAILED)
                raise ValueError(f"Workchain {workchain.pk} failed.")

            return

    def load_results(self, step: ProcessProtocol) -> None:

        # TODO: if the step is completed, we do not need to run this load_results method.
        self.load_step_data()
        
        if isinstance(step, Process):
            if isinstance(step, CommandLineTool):
                if orm.load_node(self.step_data.steps[step.uid]['workchain']).is_finished_ok:
                    step._set_outputs()
                    self.set_status(step, Status.COMPLETED)
                    self._step_completed_message(step)
                    return
            step.load_outputs()
            self._step_completed_message(step)
            return
        
        if step.prefix in ['wannier90_preproc', 'pw2wannier90']:
            self.set_status(step, Status.COMPLETED)
            return
        workchain = orm.load_node(self.step_data.steps[step.uid]['workchain'])
        if "remote_folder" in workchain.outputs:
            self.step_data.steps[step.uid]['remote_folder'] = workchain.outputs.remote_folder.pk
        output = None
        if step.ext_out == ".wout":
            output = read_output_file(step, workchain.outputs.wannier90.retrieved)
            if "remote_folder" in workchain.outputs.wannier90:
                self.step_data.steps[step.uid]['remote_folder'] = workchain.outputs.wannier90.remote_folder.pk
        elif step.ext_out in [".pwo",".w2ko",".kso",".kho",".cpo"]:
            output = read_output_file(step, workchain.outputs.retrieved)
            if hasattr(output.calc, 'kpts'):
                step.kpts = output.calc.kpts
        else:
            output = read_output_file(step, workchain.outputs.retrieved)
            
        
        if step.ext_out in [".pwo",".pro",".wout",".w2ko",".kso",".kho",".cpo",".wko"]:
            step.calc = output.calc
            step.results = output.calc.results
            #if step.ext_out == ".pwo": step.generate_band_structure() #nelec=int(workchain.outputs.output_parameters.get_dict()['number_of_electrons']))
            
        # This maybe is not the right place to do this, but we need to set the results of the step.
        if step.ext_out in [".cpo"]:
            step.results['lambda'] = step.read_ham_files()
            if step.parameters.do_bare_eigs:
                step.results['bare lambda'] = step.read_ham_files(bare=True)

        step._post_run()
        self.dump_step_data()
        self._step_completed_message(step)
        

    def load_old_calculator(self, calc: Calc):
        raise NotImplementedError # load_old_calculator(calc)
    
    def get_pseudopotential(self, library: str, element: str):

        qb = orm.QueryBuilder()
        qb.append(orm.Group, filters={'label': {'==': library}}, tag='pseudo_group')
        qb.append(UpfData, filters={'attributes.element': {'==': element}}, with_group='pseudo_group')   

        pseudo_data = None

        for pseudo in qb.all():
            with tempfile.TemporaryDirectory() as dirpath:
                temp_file = pathlib.Path(dirpath) / (pseudo[0].base.attributes.all['element'] + '.upf')
                with pseudo[0].open(pseudo[0].filename, 'rb') as handle:
                    temp_file.write_bytes(handle.read())

                pseudo_data = read_pseudo_file(temp_file)
        
        if not pseudo_data:
            raise ValueError(f"Could not find pseudopotential for element {element} in library {library}")
        
        self.step_data.pseudo_family = library
        
        return pseudo_data     
    
    def read_ham_file(self, calculator: KoopmansCPCalculator, filename: Path) -> np.ndarray[Any, np.dtype[np.complex128]]:

        new_filename = Path(str(filename).split("/")[-1])
        
        from koopmans.calculators._koopmans_cp import read_ham_file
        
        hamiltonian = None
        retrieved = orm.load_node(self.step_data.steps[calculator.uid]['workchain']).outputs.retrieved
        
        with tempfile.TemporaryDirectory() as dirpath:
            for _filename in retrieved.base.repository.list_object_names():
                if str(new_filename) == _filename:
                    # Create the file with the desired name
                    output_file = pathlib.Path(dirpath) / _filename
                    with retrieved.open(_filename, "rb") as handle:
                        output_file.write_bytes(handle.read())
                    
                    hamiltonian = read_ham_file(output_file)
        
        return hamiltonian
    
    def read_file(self, file: File, binary=False) -> str | bytes:
        if isinstance(file.parent_process, Process):
            singlefiledata = orm.load_node(self.step_data.steps[file.parent_process.uid]['input_files'][str(file.name)])
            return singlefiledata.get_content(mode='rb')
        workchain = orm.load_node(self.step_data.steps[file.parent_process.uid]['workchain'])
        filename = str(file.name).replace(file.parent_process.prefix, 'aiida').split("/")[-1]
        
        # additional replace for kc.kcw_hr_occ/emp.dat
        if "kcw_hr" in filename:
            filename = filename.replace(file.parent_process.parameters.prefix+'.', 'aiida.')
        elif "wannier90_hr" in filename:
            filename = filename.replace("wannier90_hr", 'aiida_hr')
        
        if 'wannier90' in file.parent_process.prefix:
            content =  workchain.outputs.wannier90.retrieved.get_object_content(filename, mode='r')
        else:
            mode = "rb" if ((".dat" in filename or ".xml" in filename) and binary) else "r"
            content =  workchain.outputs.retrieved.get_object_content(filename, mode=mode)
        # maybe unnecessary content post-processing
        '''content = content.split("\n")
        for line in range(len(content)):
            content[line] += "\n"
        '''
        return content
            
    def write_file(self, content: str | bytes, file: File) -> None:
        if 'inputs.pkl' in str(file.name):
            return
        
        filename = str(file.name)
        
        if isinstance(content, bytes):
            try:
                singlefile = orm.SinglefileData.from_bytes(content, filename)
            except AttributeError as e:
                import io
                singlefile = orm.SinglefileData(io.BytesIO(content), filename)
        else:
            singlefile = orm.SinglefileData.from_string(content, filename)
        singlefile.store()
        if "input_files" not in self.step_data.steps[file.parent_process.uid]:
            self.step_data.steps[file.parent_process.uid]['input_files'] = {}
        self.step_data.steps[file.parent_process.uid]['input_files'][filename] = singlefile.pk
        return singlefile
    
    def glob(self, directory: File, pattern: str, recursive: bool = False) -> Generator[File, None, None]:

        workchain = orm.load_node(self.step_data.steps[directory.parent_process.uid]['workchain'])
        if 'wannier90' in getattr(directory.parent_process, 'prefix', ''):
            listnames =  workchain.outputs.wannier90.retrieved.base.repository.list_object_names()
        else:
            listnames = workchain.outputs.retrieved.base.repository.list_object_names()
        
        for name in listnames:
            tomatch = str(directory.name / pattern)
            if hasattr(directory.parent_process, 'prefix'):
                tomatch = tomatch.replace(directory.parent_process.prefix, 'aiida')
            if isinstance(directory.parent_process, ProjwfcCalculator): # TODO: this is a hack, we need to find a better way to do this.
                tomatch = tomatch.replace(directory.parent_process.parameters.filpdos, 'aiida')
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
    
    def mkdir(self, directory: File, parents: bool = False, exist_ok: bool = False) -> None:
        """Create a directory; should mimic Path.mkdir."""
        raise NotImplementedError("Not needed for AiiDA engine")

    def rmdir(self, directory: File) -> None:
        """Remove a directory; should mimic Path.rmdir."""
        raise NotImplementedError("Not needed for AiiDA engine")
    
    def copy_file(self, source: File, destination: File, exist_ok: bool = False) -> None:
        """Copy a file; should mimic shutil.copy."""
        raise NotImplementedError("Not needed for AiiDA engine")
    
    def link_file(self, source: File, destination: File, recursive: bool = False, overwrite: bool = False) -> None:
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