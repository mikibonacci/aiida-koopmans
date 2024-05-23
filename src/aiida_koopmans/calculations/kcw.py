# -*- coding: utf-8 -*-
"""`CalcJob` implementation for the kcw.x code of Quantum ESPRESSO."""
from pathlib import Path

from aiida import orm
from aiida.plugins import DataFactory
from aiida_quantumespresso.calculations.namelists import NamelistsCalculation

SingleFileData = DataFactory('core.singlefile')

class KcwCalculation(NamelistsCalculation):
    """`CalcJob` implementation for the kcw.x code of Quantum ESPRESSO.

    kcw.x code of the Quantum ESPRESSO distribution, handles the DFPT simulations.
    For more information, refer to http://www.quantum-espresso.org/
    """

    _default_namelists = ['CONTROL','WANNIER','SCREEN','HAM']
    _blocked_keywords = [
        ('CONTROL', 'outdir', NamelistsCalculation._OUTPUT_SUBFOLDER),
        ('CONTROL', 'prefix', NamelistsCalculation._PREFIX),
        ('WANNIER', 'seedname', NamelistsCalculation._PREFIX),
    ]

    _default_parser = 'koopmans'

    xml_path = Path(NamelistsCalculation._default_parent_output_folder
                    ).joinpath(f'{NamelistsCalculation._PREFIX}.save', 'data-file-schema.xml')
    _internal_retrieve_list = [
        NamelistsCalculation._PREFIX + '.pdos*',
    ]
    # The XML file is added to the temporary retrieve list since it is required for parsing, but already in the
    # repository of a an ancestor calculation.
    _retrieve_temporary_list = [
        xml_path.as_posix(),
    ]

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        from aiida.orm import BandsData, ProjectionData
        super().define(spec)
        spec.input('parent_folder', valid_type=(orm.RemoteData, orm.FolderData), help='The output folder of a pw.x calculation')
        spec.input('kpoints', valid_type=orm.KpointsData, help='kpoint path if do_bands=True in the parameters', required=False)
        #spec.input('wann_occ_hr', valid_type=SingleFileData, help='wann_occ_hr', required=False)
        #spec.input('wann_emp_hr', valid_type=SingleFileData, help='wann_emp_hr', required=False)
        spec.input('alpha_occ', valid_type=SingleFileData, help='alpha_occ', required=False)
        spec.input('alpha_emp', valid_type=SingleFileData, help='alpha_emp', required=False)
        spec.input('wann_u_mat', valid_type=SingleFileData, help='wann_occ_u', required=False)
        spec.input('wann_emp_u_mat', valid_type=SingleFileData, help='wann_emp_u', required=False)
        spec.input('wann_emp_u_dis_mat', valid_type=SingleFileData, help='wann_dis_u', required=False)
        spec.input('wann_centres_xyz', valid_type=SingleFileData, help='wann_occ_centres', required=False)
        spec.input('wann_emp_centres_xyz', valid_type=SingleFileData, help='wann_emp_centres', required=False)
        spec.input('settings', valid_type=orm.Dict, required=True, default=lambda: orm.Dict({
            'CMDLINE': ["-in", cls._DEFAULT_INPUT_FILE],
            }), help='Use an additional node for special settings',) #validator=validate_parameters,)

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

    def prepare_for_submission(self, folder):
        calcinfo = super().prepare_for_submission(folder)

        for wann_file in ['wann_u_mat','wann_emp_u_mat','wann_emp_u_dis_mat','wann_centres_xyz','wann_emp_centres_xyz']:
            if hasattr(self.inputs,wann_file):
                wannier_singelfiledata = getattr(self.inputs, wann_file)
                calcinfo.local_copy_list.append((wannier_singelfiledata.uuid, wannier_singelfiledata.filename, wann_file.replace("_mat",".mat").replace("_xyz",".xyz").replace("wann","aiida")))

        for alpha_file in ['alpha_occ','alpha_emp']:
            if hasattr(self.inputs,alpha_file):
                suffix = alpha_file.replace("alpha_occ","").replace("alpha_emp","_empty")
                alpha_singelfiledata = getattr(self.inputs, alpha_file)
                calcinfo.local_copy_list.append((alpha_singelfiledata.uuid, alpha_singelfiledata.filename,f'file_alpharef{suffix}.txt'))


        if hasattr(self.inputs,"kpoints"):
            kpoints_card = prepare_kpoints_card(self.inputs.kpoints)
            with folder.open(self.metadata.options.input_filename, 'a+') as handle:
                handle.write(kpoints_card)
                
        return calcinfo
    
def prepare_kpoints_card(kpoints=None):
    # from the BasePwCpInputGenerator, I had to move it here as we cannot just inherit
    from aiida.common import exceptions
    # ============ I prepare the k-points =============
    kpoints_card = ''

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
                        'At least one k point must be provided for non-gamma calculations'
                    ) from exception
            except AttributeError:
                raise exceptions.InputValidationError('No valid kpoints have been found') from exception

            try:
                _, weights = kpoints.get_kpoints(also_weights=True)
            except AttributeError:
                weights = [1.] * num_kpoints

        gamma_only = False # settings.pop('GAMMA_ONLY', False)

        if gamma_only:
            if has_mesh:
                if tuple(mesh) != (1, 1, 1) or tuple(offset) != (0., 0., 0.):
                    raise exceptions.InputValidationError(
                        'If a gamma_only calculation is requested, the '
                        'kpoint mesh must be (1,1,1),offset=(0.,0.,0.)'
                    )

            else:
                if (len(kpoints_list) != 1 or tuple(kpoints_list[0]) != tuple(0., 0., 0.)):
                    raise exceptions.InputValidationError(
                        'If a gamma_only calculation is requested, the '
                        'kpoints coordinates must only be (0.,0.,0.)'
                    )

            kpoints_type = 'gamma'

        elif has_mesh:
            kpoints_type = 'automatic'

        else:
            kpoints_type = 'crystal'

        kpoints_card_list = [f'K_POINTS {kpoints_type}\n']

        if kpoints_type == 'automatic':
            if any(i not in [0, 0.5] for i in offset):
                raise exceptions.InputValidationError('offset list must only be made of 0 or 0.5 floats')
            the_offset = [0 if i == 0. else 1 for i in offset]
            the_6_integers = list(mesh) + the_offset
            kpoints_card_list.append('{:d} {:d} {:d} {:d} {:d} {:d}\n'.format(*the_6_integers))  # pylint: disable=consider-using-f-string

        elif kpoints_type == 'gamma':
            # nothing to be written in this case
            pass
        else:
            kpoints_card_list.append(f'{num_kpoints:d}\n')
            for kpoint, weight in zip(kpoints_list, weights):
                kpoints_card_list.append(
                    f'  {kpoint[0]:18.10f} {kpoint[1]:18.10f} {kpoint[2]:18.10f} {weight:18.10f}\n'
                )

        kpoints_card = ''.join(kpoints_card_list)
        del kpoints_card_list
        return kpoints_card