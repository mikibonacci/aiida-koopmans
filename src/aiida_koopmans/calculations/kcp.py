# -*- coding: utf-8 -*-
"""`CalcJob` implementation for the kcw.x code of Quantum ESPRESSO."""
from pathlib import Path
import os

from aiida import orm
from aiida.plugins import DataFactory
from aiida_quantumespresso.calculations.cp import CpCalculation

class KcpCalculation(CpCalculation):
    
    _automatic_namelists = {
        'cp': ['CONTROL', 'SYSTEM', 'ELECTRONS', 'IONS', 'EE', 'CELL', 'NKSIC']
    }

    xml_path = Path(CpCalculation._OUTPUT_SUBFOLDER
                    ).joinpath(f'{CpCalculation._PREFIX}_{CpCalculation._CP_WRITE_UNIT_NUMBER}.save', 
                               'data-file.xml')
    
    _retrieve_temporary_list = [
        xml_path.as_posix(),
    ]
    
    _default_symlink_usage = False
    
    # Pieces of input that we won't allow users to set
    _blocked_keywords = [
        ('CONTROL', 'pseudo_dir'),  # set later
        ('CONTROL', 'outdir'),  # set later
        ('CONTROL', 'prefix'),  # set later
        ('SYSTEM', 'celldm'),
        ('SYSTEM', 'nat'),  # set later
        ('SYSTEM', 'ntyp'),  # set later
        ('SYSTEM', 'a'),
        ('SYSTEM', 'b'),
        ('SYSTEM', 'c'),
        ('SYSTEM', 'cosab'),
        ('SYSTEM', 'cosac'),
        ('SYSTEM', 'cosbc'),
        #('CONTROL', 'ndr', _CP_READ_UNIT_NUMBER),
        #('CONTROL', 'ndw', _CP_WRITE_UNIT_NUMBER),
    ]
    
    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.input('metadata.options.parser_name', valid_type=str, default='koopmans.kcp')
        spec.input('spin2_files', valid_type=orm.List, required=False)
        spec.input('file_alpharef', valid_type=(orm.SinglefileData, orm.RemoteData), help='alpha occupied', required=False)
        spec.input('file_alpharef_empty', valid_type=(orm.SinglefileData, orm.RemoteData), help='alpha empty', required=False)
        spec.input('additional_files_K0001', valid_type=orm.Dict, help='additional files', required=False)
        spec.input('additional_files_wann2kcp', valid_type=orm.Dict, help='additional files', required=False)
        spec.input('retrieve_dat_files', valid_type=orm.Bool, default=lambda: orm.Bool(False), help='retrieve dat files', required=False)
                   
        spec.output('output_trajectory', valid_type=orm.TrajectoryData)
        spec.output('output_parameters', valid_type=orm.Dict)
        spec.default_output_node = 'output_parameters'

        spec.exit_code(301, 'ERROR_NO_RETRIEVED_TEMPORARY_FOLDER',
            message='The retrieved temporary folder could not be accessed.')
        spec.exit_code(303, 'ERROR_MISSING_XML_FILE',
            message='The required XML file is not present in the retrieved folder.')
        spec.exit_code(304, 'ERROR_OUTPUT_XML_MULTIPLE',
            message='The retrieved folder contains multiple XML files.')
        spec.exit_code(320, 'ERROR_OUTPUT_XML_READ',
            message='The required XML file could not be read.')
        spec.exit_code(330, 'ERROR_READING_POS_FILE',
            message='The required POS file could not be read.')
        spec.exit_code(340, 'ERROR_READING_TRAJECTORY_DATA',
            message='The required trajectory data could not be read.')
        # yapf: enable
        
    def prepare_for_submission(self, folder):
        
        write_unit_number = self.inputs.parameters.get_dict()["control"].get("ndw",CpCalculation._CP_WRITE_UNIT_NUMBER)
        read_unit_number = self.inputs.parameters.get_dict()["control"].get("ndr",CpCalculation._CP_READ_UNIT_NUMBER)
        
        if 'parent_folder' in self.inputs:
            old_nd = self.inputs.parent_folder.creator.inputs.parameters.get_dict()["control"].get("ndw",read_unit_number) # ndw of the parent calculation, which should be the same as ndr of the current calculation
                        
            self._restart_copy_from = str(Path(self._OUTPUT_SUBFOLDER).joinpath(f'{CpCalculation._PREFIX}_{read_unit_number}.save')) if old_nd == read_unit_number else str(Path(self._OUTPUT_SUBFOLDER).joinpath(f'{CpCalculation._PREFIX}_{old_nd}.save'))
            self._restart_copy_to = str(Path(self._OUTPUT_SUBFOLDER)) if old_nd == read_unit_number else str(Path(self._OUTPUT_SUBFOLDER).joinpath(f'{CpCalculation._PREFIX}_{read_unit_number}.save'))

        calcinfo = super().prepare_for_submission(folder)

        if getattr(self.inputs, 'retrieve_dat_files', False):
            calcinfo.retrieve_list.append(str(Path(self._OUTPUT_SUBFOLDER).joinpath(f'{CpCalculation._PREFIX}_{write_unit_number}.save','K00001', 
                                '*.dat')))
        
        calcinfo.retrieve_list.append(str(Path(self._OUTPUT_SUBFOLDER).joinpath(f'{CpCalculation._PREFIX}_{write_unit_number}.save','K00001', 
                               '*.xml')))
        calcinfo.retrieve_list.append("file_alpha*")
        calcinfo.retrieve_list.append("*.out")
        calcinfo.retrieve_list.append("ham_*")
        
        
        # TODO: generalize all the paths, using Path...
        if 'spin2_files' in self.inputs:
            for name, file_pk in self.inputs.spin2_files.get_list():
                file = orm.load_node(file_pk)
                calcinfo.local_copy_list.append((file.uuid, file.filename, str(Path(self._OUTPUT_SUBFOLDER).joinpath(f'{CpCalculation._PREFIX}_{read_unit_number}.save','K00001',file.filename))))
        
        if "additional_files_K0001" in self.inputs:
            for calc_pk, file_list in self.inputs.additional_files_K0001.get_dict().items():
                calc = orm.load_node(calc_pk)
                calc_read_unit_number = calc.inputs.parameters.get_dict()["control"].get("ndw",CpCalculation._CP_READ_UNIT_NUMBER)
                remote_folder = calc.outputs.remote_folder
                for file_name in file_list:
                    calcinfo.remote_copy_list.append((
                        remote_folder.computer.uuid, 
                        remote_folder.get_remote_path()+'/'+str(Path(self._OUTPUT_SUBFOLDER).joinpath(f'{CpCalculation._PREFIX}_{calc_read_unit_number}.save','K00001',file_name.replace("_occupied","fixed_empty"))),
                        './'+str(Path(self._OUTPUT_SUBFOLDER).joinpath(f'{CpCalculation._PREFIX}_{read_unit_number}.save','K00001',file_name)),
                    ))
                                   
        if "additional_files_wann2kcp" in self.inputs: 
            for calc_pk, file_list in self.inputs.additional_files_wann2kcp.get_dict().items():
                calc = orm.load_node(calc_pk)
                remote_folder = calc.outputs.remote_folder
                for file_name in file_list:
                    calcinfo.remote_copy_list.append((
                        remote_folder.computer.uuid, 
                        remote_folder.get_remote_path()+'/'+file_name[1],
                        './'+str(Path(self._OUTPUT_SUBFOLDER).joinpath(f'{CpCalculation._PREFIX}_{read_unit_number}.save','K00001',file_name[0])),
                    )) 
                    
        if "file_alpharef" in self.inputs:
            file = self.inputs.file_alpharef
            calcinfo.local_copy_list.append((file.uuid, file.filename, file.filename))
        if "file_alpharef_empty" in self.inputs:
            file = self.inputs.file_alpharef_empty
            calcinfo.local_copy_list.append((file.uuid, file.filename, file.filename))
        
        return calcinfo
