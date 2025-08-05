"""`CalcJob` implementation for the kcw.x code of Quantum ESPRESSO."""
import os
from pathlib import Path
from typing import Any

from aiida_quantumespresso.calculations.namelists import NamelistsCalculation

from aiida import orm
from aiida.orm import BandsData


class KcwCalculation(NamelistsCalculation):
    """`CalcJob` implementation for the kcw.x code of Quantum ESPRESSO.

    kcw.x code of the Quantum ESPRESSO distribution, handles the DFPT simulations.
    For more information, refer to http://www.quantum-espresso.org/
    """

    _default_namelists = ["CONTROL", "WANNIER", "SCREEN", "HAM"]
    _blocked_keywords = [
        ("CONTROL", "outdir", NamelistsCalculation._OUTPUT_SUBFOLDER),
        ("CONTROL", "prefix", NamelistsCalculation._PREFIX),
        ("WANNIER", "seedname", NamelistsCalculation._PREFIX),
    ]

    _default_parser = "koopmans.kcw"

    xml_path = Path(NamelistsCalculation._default_parent_output_folder).joinpath(
        f"{NamelistsCalculation._PREFIX}.save", "data-file-schema.xml"
    )
    _internal_retrieve_list = ["*.dat"]
    # The XML file is added to the temporary retrieve list since it is required for parsing, but already in the
    # repository of a an ancestor calculation.
    _retrieve_temporary_list = [
        xml_path.as_posix(),
    ]

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.input('parent_folder', valid_type=(
            orm.RemoteData, orm.FolderData), help='The output folder of a pw.x calculation')
        spec.input('kpoints', valid_type=orm.KpointsData,
                   help='kpoint path if do_bands=True in the parameters', required=False)
        # spec.input('wann_occ_hr', valid_type=SingleFileData, help='wann_occ_hr', required=False)
        # spec.input('wann_emp_hr', valid_type=SingleFileData, help='wann_emp_hr', required=False)
        spec.input('alpha', valid_type=(orm.SinglefileData,
                   orm.RemoteData), help='alpha', required=False)
        spec.input('wann_u_mat', valid_type=(orm.SinglefileData, orm.RemoteData),
                   help='Rotation matrix file `<seedname>_emp_u.mat` for when `l_unique_manifold` is `True`', required=False)
        spec.input('wann_occ_u_mat', valid_type=(orm.SinglefileData, orm.RemoteData),
                   help='Rotation matrix file for the occupied manifold `<seedname>_u.mat` for when `l_unique_manifold` is `False`', required=False)
        spec.input('wann_emp_u_mat', valid_type=(orm.SinglefileData, orm.RemoteData),
                   help='Rotation matrix file for the empty manifold `<seedname>_emp_u.mat` for when `l_unique_manifold` is `False`', required=False)
        spec.input('wann_u_dis_mat', valid_type=(orm.SinglefileData, orm.RemoteData),
                   help='Disentanglement matrix file `<seedname>_u_dis.mat` for when `l_unique_manifold` is `True`', required=False)
        spec.input('wann_emp_u_dis_mat', valid_type=(orm.SinglefileData, orm.RemoteData),
                   help='Disentanglement matrix file `<seedname>_emp_u_dis.mat` for when `l_unique_manifold` is `False`', required=False)
        spec.input('wann_centres_xyz', valid_type=(orm.SinglefileData, orm.RemoteData),
                   help='Wannier centres file `<seedname>_centres.xyz` for when `l_unique_manifold` is `True`', required=False)
        spec.input('wann_occ_centres_xyz', valid_type=(orm.SinglefileData, orm.RemoteData),
                   help='Wannier centres file for the occupied manifold `<seedname>_centres.xyz` for when `l_unique_manifold` is `False`', required=False)
        spec.input('wann_emp_centres_xyz', valid_type=(orm.SinglefileData, orm.RemoteData),
                   help='Wannier centres file for the empty manifold `<seedname>_emp_centres.xyz` for when `l_unique_manifold` is `False`', required=False)
        spec.input('settings', valid_type=orm.Dict, required=True, default=lambda: orm.Dict({
            'CMDLINE': ["-in", cls._DEFAULT_INPUT_FILE],
        }), help='Use an additional node for special settings',)  # validator=validate_parameters,)

        spec.inputs.validator = cls.validate_inputs

        spec.output('output_parameters', valid_type=orm.Dict, required=False)
        spec.output('bands', valid_type=BandsData, required=False)
        spec.default_output_node = 'output_parameters'

        spec.exit_code(301, 'ERROR_NO_RETRIEVED_TEMPORARY_FOLDER',
                       message='The retrieved temporary folder could not be accessed.')
        spec.exit_code(303, 'ERROR_OUTPUT_XML_MISSING',
                       message='The retrieved folder did not contain the required XML file.')
        spec.exit_code(320, 'ERROR_OUTPUT_XML_READ',
                       message='The XML output file could not be read.')
        spec.exit_code(321, 'ERROR_OUTPUT_XML_PARSE',
                       message='The XML output file could not be parsed.')
        spec.exit_code(322, 'ERROR_OUTPUT_XML_FORMAT',
                       message='The XML output file has an unsupported format.')
        spec.exit_code(330, 'ERROR_READING_PDOSTOT_FILE',
                       message='The pdos_tot file could not be read from the retrieved folder.')
        spec.exit_code(340, 'ERROR_PARSING_PROJECTIONS',
                       message='An exception was raised parsing bands and projections.')
        # yapf: enable

    @classmethod
    def validate_inputs(cls, value: dict[str, Any], _):
        """Validate the inputs.

        Specifically, check if the provided Wannier90 files are consistent with the `l_unique_manifold` parameter.
        """

        parameters = value["parameters"].get_dict()
        l_unique_manifold: bool = parameters.get("WANNIER", {}).get(
            "l_unique_manifold", False
        )
        assert isinstance(l_unique_manifold, bool)

        for file in value.keys():
            if not file.startswith("wann_"):
                continue

            required = l_unique_manifold ^ ("occ" in file or "emp" in file)

            if required and value.get(file, None) is None:
                raise ValueError(
                    f"`{file}` is required when `l_unique_manifold` is `{l_unique_manifold}`."
                )
            elif not required and value.get(file, None) is not None:
                raise ValueError(
                    f"`{file}` is not valid when `l_unique_manifold` is `{l_unique_manifold}`."
                )

        return

    def prepare_for_submission(self, folder):
        calcinfo = super().prepare_for_submission(folder)

        # Access the wannier90 files
        for ext in ["u_mat", "u_dis_mat", "centres_xyz"]:
            for occ_str in ["", "_occ", "_emp"]:
                wannier_input_name = f"wann{occ_str}_{ext}"
                wannier_input_file = getattr(self.inputs, wannier_input_name, None)

                if wannier_input_file is None:
                    continue

                src_name = wannier_input_file.filename
                dst_name = (
                    wannier_input_name.replace("_mat", ".mat")
                    .replace("_xyz", ".xyz")
                    .replace("wann", "aiida")
                    .replace("_occ", "")
                )

                if isinstance(wannier_input_file, orm.SinglefileData):
                    # Copy the local file to the remote
                    calcinfo.local_copy_list.append(
                        (wannier_input_file.uuid, src_name, dst_name)
                    )
                elif isinstance(wannier_input_file, orm.RemoteData):
                    # Symlink the remote file
                    calcinfo.remote_symlink_list.append(
                        create_symlink_tuple(
                            parent_folder=wannier_input_file,
                            filename=src_name,
                            target=dst_name,
                        )
                    )

        if hasattr(self.inputs, "alpha"):
            alpha_singlefiledata = getattr(self.inputs, "alpha")
            calcinfo.local_copy_list.append(
                (
                    alpha_singlefiledata.uuid,
                    alpha_singlefiledata.filename,
                    "file_alpharef.txt",
                )
            )

        if hasattr(self.inputs, "kpoints"):
            kpoints_card = prepare_kpoints_card(self.inputs.kpoints)
            with folder.open(self.metadata.options.input_filename, "a+") as handle:
                handle.write(kpoints_card)

        del calcinfo.codes_info[0].stdin_name
        calcinfo.codes_info[0].cmdline_params = self.inputs.settings.get("CMDLINE", [])
        if "-in" not in calcinfo.codes_info[0].cmdline_params:
            calcinfo.codes_info[0].cmdline_params.append("-in")
        if (
            self.inputs.metadata.options.input_filename
            not in calcinfo.codes_info[0].cmdline_params
        ):
            calcinfo.codes_info[0].cmdline_params.append(
                self.metadata.options.input_filename
            )

        return calcinfo


def create_symlink_tuple(parent_folder: orm.RemoteData, filename: str, target: str):
    return (
        parent_folder.computer.uuid,
        os.path.join(parent_folder.get_remote_path(), filename),
        target,
    )


def prepare_kpoints_card(kpoints=None):
    # from the BasePwCpInputGenerator, I had to move it here as we cannot just inherit
    from aiida.common import exceptions

    # ============ I prepare the k-points =============
    kpoints_card = ""

    if kpoints:
        try:
            mesh, offset = kpoints.get_kpoints_mesh()
            has_mesh = True
            """force_kpoints_list = settings.pop('FORCE_KPOINTS_LIST', False)
            if force_kpoints_list:
                kpoints_list = kpoints.get_kpoints_mesh(print_list=True)
                num_kpoints = len(kpoints_list)
                has_mesh = False
                weights = [1.] * num_kpoints
            """

        except AttributeError as exception:

            try:
                kpoints_list = kpoints.get_kpoints()
                num_kpoints = len(kpoints_list)
                has_mesh = False
                if num_kpoints == 0:
                    raise exceptions.InputValidationError(
                        "At least one k point must be provided for non-gamma calculations"
                    ) from exception
            except AttributeError:
                raise exceptions.InputValidationError(
                    "No valid kpoints have been found"
                ) from exception

            try:
                _, weights = kpoints.get_kpoints(also_weights=True)
            except AttributeError:
                weights = [1.0] * num_kpoints

        gamma_only = False  # settings.pop('GAMMA_ONLY', False)

        if gamma_only:
            if has_mesh:
                if tuple(mesh) != (1, 1, 1) or tuple(offset) != (0.0, 0.0, 0.0):
                    raise exceptions.InputValidationError(
                        "If a gamma_only calculation is requested, the "
                        "kpoint mesh must be (1,1,1),offset=(0.,0.,0.)"
                    )

            else:
                if len(kpoints_list) != 1 or tuple(kpoints_list[0]) != tuple(
                    0.0, 0.0, 0.0
                ):
                    raise exceptions.InputValidationError(
                        "If a gamma_only calculation is requested, the "
                        "kpoints coordinates must only be (0.,0.,0.)"
                    )

            kpoints_type = "gamma"

        elif has_mesh:
            kpoints_type = "automatic"

        else:
            kpoints_type = "crystal"

        kpoints_card_list = [f"K_POINTS {kpoints_type}\n"]

        if kpoints_type == "automatic":
            if any(i not in [0, 0.5] for i in offset):
                raise exceptions.InputValidationError(
                    "offset list must only be made of 0 or 0.5 floats"
                )
            the_offset = [0 if i == 0.0 else 1 for i in offset]
            the_6_integers = list(mesh) + the_offset
            kpoints_card_list.append(
                "{:d} {:d} {:d} {:d} {:d} {:d}\n".format(*the_6_integers)
            )  # pylint: disable=consider-using-f-string

        elif kpoints_type == "gamma":
            # nothing to be written in this case
            pass
        else:
            kpoints_card_list.append(f"{num_kpoints:d}\n")
            for kpoint, weight in zip(kpoints_list, weights):
                kpoints_card_list.append(
                    f"  {kpoint[0]:18.10f} {kpoint[1]:18.10f} {kpoint[2]:18.10f} {weight:18.10f}\n"
                )

        kpoints_card = "".join(kpoints_card_list)
        del kpoints_card_list
        return kpoints_card
