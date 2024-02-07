# -*- coding: utf-8 -*-
"""`CalcJob` implementation for the kcw.x code of Quantum ESPRESSO."""
from pathlib import Path

from aiida import orm

from aiida_quantumespresso.calculations.namelists import NamelistsCalculation

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