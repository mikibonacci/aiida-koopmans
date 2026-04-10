# -*- coding: utf-8 -*-
"""`CalcJob` implementation for the kcw.x code of Quantum ESPRESSO."""
from pathlib import Path
import os

from aiida import orm
from aiida.plugins import DataFactory
from aiida.common import datastructures, exceptions
from aiida.common.warnings import AiidaDeprecationWarning

from aiida_quantumespresso.calculations import _uppercase_dict, _case_transform_dict


def _lowercase_dict(dictionary, dict_name):
    return _case_transform_dict(dictionary, dict_name, '_lowercase_dict', str.lower)
from aiida_quantumespresso.utils.convert import convert_input_to_namelist_entry
from aiida_quantumespresso.calculations.pp import PpCalculation


class Wann2kcpCalculation(PpCalculation):
    """`CalcJob` implementation for the wann2kcp.x code of Quantum ESPRESSO.
    """
    
    _default_parser = 'koopmans.wann2kcp'
    
    # Default name of the subfolder inside 'parent_folder' from which you want to copy the files, in case the
    # parent_folder is of type FolderData
    _INPUT_SUBFOLDER = './out/'

    # In the PW Calculation plugin, these folder names and prefixes are fixed, so import them to reduce maintenance
    # pylint: disable=protected-access
    from aiida_quantumespresso.calculations import BasePwCpInputGenerator
    _OUTPUT_SUBFOLDER = BasePwCpInputGenerator._OUTPUT_SUBFOLDER
    _PREFIX = BasePwCpInputGenerator._PREFIX
    _PSEUDO_SUBFOLDER = BasePwCpInputGenerator._PSEUDO_SUBFOLDER
    _DEFAULT_INPUT_FILE = BasePwCpInputGenerator._DEFAULT_INPUT_FILE
    _DEFAULT_OUTPUT_FILE = BasePwCpInputGenerator._DEFAULT_OUTPUT_FILE
    # pylint: enable=protected-access

    # Grid data output file from first stage of pp calculation
    _FILPLOT = 'aiida.filplot'
    # Grid data output in desired format
    _FILEOUT = 'aiida.fileout'
    
    _default_namelists = ['INPUTPP']

    # Keywords that cannot be set by the user but will be set by the plugin
    _blocked_keywords = [
        ('INPUTPP', 'outdir', _OUTPUT_SUBFOLDER), 
        ('INPUTPP', 'prefix', _PREFIX),
        ('INPUTPP', 'seedname', _PREFIX),
    ]
    
    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.input('metadata.options.parser_name', valid_type=str, default='koopmans.wann2kcp')
        spec.input('parent_folder', valid_type=(orm.RemoteData, orm.FolderData), required=True,
            help='Output folder of a completed `PwCalculation`')
        spec.input('parameters', valid_type=orm.Dict, required=True,
            help='Use a node that specifies the input parameters for the namelists')
        spec.input('additional_files_wannier', valid_type=orm.Dict, help='additional files', required=False)

    def prepare_for_submission(self, folder):

        parameters = _uppercase_dict(self.inputs.parameters.get_dict(), dict_name='parameters')
        parameters = {k: _lowercase_dict(v, dict_name=k) for k, v in parameters.items()}

        # Same for settings.
        if 'settings' in self.inputs:
            settings = _uppercase_dict(self.inputs.settings.get_dict(), dict_name='settings')
        else:
            settings = {}

        # Set default values. NOTE: this is different from PW/CP
        for blocked in self._blocked_keywords:
            namelist = blocked[0].upper()
            key = blocked[1].lower()
            value = blocked[2]

            if namelist in parameters:
                if key in parameters[namelist]:
                    raise exceptions.InputValidationError(
                        f"You cannot specify explicitly the '{key}' key in the '{namelist}' namelist."
                    )
            else:
                parameters[namelist] = {}
            parameters[namelist][key] = value

        # Restrict the plot output to the file types that we want to be able to parse
        dimension_to_output_format = {
            0: 0,  # Spherical integration -> Gnuplot, 1D
            1: 0,  # 1D -> Gnuplot, 1D
            2: 7,  # 2D -> Gnuplot, 2D
            3: 6,  # 3D -> Gaussian cube
            4: 0,  # Polar on a sphere -> # Gnuplot, 1D
        }
        
        # NOTE: I commented this out because it is not needed.
        # NOTE: I basically copied the PpCalculation class and modified. Should not be like this probably.
        #parameters['PLOT']['output_format'] = dimension_to_output_format[parameters['PLOT']['iflag']]

        namelists_toprint = self._default_namelists

        input_filename = self.inputs.metadata.options.input_filename
        with folder.open(input_filename, 'w') as infile:
            for namelist_name in namelists_toprint:
                infile.write(f'&{namelist_name}\n')
                # namelist content; set to {} if not present, so that we leave an empty namelist
                namelist = parameters.pop(namelist_name, {})
                for key, value in sorted(namelist.items()):
                    infile.write(convert_input_to_namelist_entry(key, value))
                infile.write('/\n')

        # Check for specified namelists that are not expected
        if parameters:
            raise exceptions.InputValidationError(
                'The following namelists are specified in parameters, but are not valid namelists for the current type '
                f'of calculation: {",".join(list(parameters.keys()))}'
            )

        remote_copy_list = []
        local_copy_list = []

        source = self.inputs.get('parent_folder', None)

        if isinstance(source, orm.RemoteData):
            dirpath = os.path.join(source.get_remote_path(), self._INPUT_SUBFOLDER)
            remote_copy_list.append((source.computer.uuid, dirpath, self._OUTPUT_SUBFOLDER))
            dirpath = os.path.join(source.get_remote_path(), self._PSEUDO_SUBFOLDER)
            remote_copy_list.append((source.computer.uuid, dirpath, self._PSEUDO_SUBFOLDER))
        elif isinstance(source, orm.FolderData):
            local_copy_list.append((source.uuid, self._OUTPUT_SUBFOLDER, self._OUTPUT_SUBFOLDER))
            local_copy_list.append((source.uuid, self._PSEUDO_SUBFOLDER, self._PSEUDO_SUBFOLDER))

        codeinfo = datastructures.CodeInfo()
        codeinfo.cmdline_params = settings.pop('CMDLINE', [])
        codeinfo.stdin_name = self.inputs.metadata.options.input_filename
        codeinfo.stdout_name = self.inputs.metadata.options.output_filename
        codeinfo.code_uuid = self.inputs.code.uuid

        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = local_copy_list
        calcinfo.remote_copy_list = remote_copy_list

        # Retrieve by default the output file
        calcinfo.retrieve_list = [self.inputs.metadata.options.output_filename]
        calcinfo.retrieve_temporary_list = []

        # Depending on the `plot_num` and the corresponding parameters, more than one pair of `filplot` + `fileout`
        # files may be written. In that case, the data files will have `filplot` as a prefix with some suffix to
        # distinguish them from one another. The `fileout` filename will be the full data filename with the `fileout`
        # value as a suffix.
        retrieve_tuples = [self._FILEOUT, (f'{self._FILPLOT}_*{self._FILEOUT}', '.', 0)]
        if 'keep_plot_file' in self.inputs.metadata.options:
            self.inputs.metadata.options.keep_data_files = self.inputs.metadata.options.keep_plot_file
            warnings.warn(
                "The input parameter 'keep_plot_file' is deprecated and will be removed in version 5.0.0. "
                "Please use 'keep_data_files' instead.", AiidaDeprecationWarning
            )
        if self.inputs.metadata.options.keep_data_files:
            calcinfo.retrieve_list.extend(retrieve_tuples)
        # If we do not want to parse the retrieved files, temporary retrieval is meaningless
        elif self.inputs.metadata.options.parse_data_files:
            calcinfo.retrieve_temporary_list.extend(retrieve_tuples)

        ### NEW ###
        if "additional_files_wannier" in self.inputs:
            for remote_folder_pk, file_list in self.inputs.additional_files_wannier.get_dict().items():
                remote_folder = orm.load_node(remote_folder_pk)
                for file_name in file_list:
                    calcinfo.remote_copy_list.append((
                        remote_folder.computer.uuid, 
                        remote_folder.get_remote_path()+'/'+str(file_name.replace("wannier90","aiida")),
                        './'+str(file_name.replace("wannier90","aiida"))))
        
        return calcinfo
                                    