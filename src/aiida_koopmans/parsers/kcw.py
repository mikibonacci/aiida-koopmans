# -*- coding: utf-8 -*-
from pathlib import Path
import pathlib
import tempfile
from ase import io

from aiida.orm import Dict

from aiida_quantumespresso.parsers.parse_raw.base import convert_qe_to_aiida_structure, convert_qe_to_kpoints
from aiida_quantumespresso.utils.mapping import get_logging_container

from aiida_quantumespresso.parsers.base import BaseParser

class KcwParser(BaseParser):
    """``Parser`` implementation for the ``KcwCalculation`` calculation job class.

    For now it just checks if there is the `JOB DONE` string at the end, and some other common functionalities (BaseParser).
    """
    
    def parse(self, **kwargs):
        """Parse the retrieved files from a ``KcwCalculation`` into output nodes."""
        # we create a dictionary the progressively accumulates more info
        out_info_dict = {}

        logs = get_logging_container()

        stdout, parsed_data, logs = self.parse_stdout_from_retrieved(logs)
        out_info_dict['out_file'] = stdout.split('\n')

        base_exit_code = self.check_base_errors(logs)
        if base_exit_code:
            return self.exit(base_exit_code, logs)
        
        # Create temporary directory. However, see aiida-wannier90-workflows/src/aiida_wannier90_workflows/utils/workflows/pw.py for more advanced and smart ways.
        retrieved = self.retrieved
        with tempfile.TemporaryDirectory() as dirpath:
            # Open the output file from the AiiDA storage and copy content to the temporary file
            for filename in retrieved.base.repository.list_object_names():
                if '.out' in filename:
                    # Create the file with the desired name
                    readable_filename = "kc.kho"
                    temp_file = pathlib.Path(dirpath) / readable_filename
                    with retrieved.open(filename, 'rb') as handle:
                        temp_file.write_bytes(handle.read())
                
                    output = io.read(temp_file)
                    
        if "eigenvalues" in output.calc.results.keys():
            parsed_data["eigenvalues"] = output.calc.results["eigenvalues"]

        self.out('output_parameters', Dict(parsed_data))

        if 'ERROR_OUTPUT_STDOUT_INCOMPLETE'in logs.error:
            return self.exit(self.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE, logs)

        try:
            retrieved_temporary_folder = kwargs['retrieved_temporary_folder']
        except KeyError:
            return self.exit(self.exit_codes.ERROR_NO_RETRIEVED_TEMPORARY_FOLDER, logs)

        # Parse the XML to obtain the `structure`, `kpoints` and spin-related settings from the parent calculation
        self.exit_code_xml = None
        parsed_xml, logs_xml = self._parse_xml(retrieved_temporary_folder)
        self.emit_logs(logs_xml)

        if self.exit_code_xml:
            return self.exit(self.exit_code_xml)

        out_info_dict['structure'] = convert_qe_to_aiida_structure(parsed_xml['structure'])
        out_info_dict['kpoints'] = convert_qe_to_kpoints(parsed_xml, out_info_dict['structure'])
        out_info_dict['nspin'] = parsed_xml.get('number_of_spin_components')
        out_info_dict['collinear'] = not parsed_xml.get('non_colinear_calculation')
        out_info_dict['spinorbit'] = parsed_xml.get('spin_orbit_calculation')
        out_info_dict['spin'] = out_info_dict['nspin'] == 2

        # check and read alpha file: use the `koopmans` package capabilities.
        """
        out_filenames = self.retrieved.base.repository.list_object_names()
        try:
            pdostot_filename = fnmatch.filter(out_filenames, '*pdos_tot*')[0]
            with self.retrieved.base.repository.open(pdostot_filename, 'r') as pdostot_file:
                # Columns: Energy(eV), Ldos, Pdos
                pdostot_array = np.atleast_2d(np.genfromtxt(pdostot_file))
                energy = pdostot_array[:, 0]
                dos = pdostot_array[:, 1]
        except (OSError, KeyError):
            return self.exit(self.exit_codes.ERROR_READING_PDOSTOT_FILE, logs)
        """


        return self.exit(logs=logs)

    def _parse_xml(self, retrieved_temporary_folder):
        """Parse the XML file.

        The XML must be parsed in order to obtain the required information for the orbital parsing.
        """
        from aiida_quantumespresso.parsers.parse_xml.exceptions import XMLParseError, XMLUnsupportedFormatError
        from aiida_quantumespresso.parsers.parse_xml.pw.parse import parse_xml

        logs = get_logging_container()
        parsed_xml = {}

        xml_filepath = Path(retrieved_temporary_folder) / self.node.process_class.xml_path.name

        if not xml_filepath.exists():
            self.exit_code_xml = self.exit_codes.ERROR_OUTPUT_XML_MISSING
            return parsed_xml, logs

        try:
            with xml_filepath.open('r') as handle:
                parsed_xml, logs = parse_xml(handle, None)
        except IOError:
            self.exit_code_xml = self.exit_codes.ERROR_OUTPUT_XML_READ
        except XMLParseError:
            self.exit_code_xml = self.exit_codes.ERROR_OUTPUT_XML_PARSE
        except XMLUnsupportedFormatError:
            self.exit_code_xml = self.exit_codes.ERROR_OUTPUT_XML_FORMAT
        except Exception as exc:
            self.exit_code_xml = self.exit_codes.ERROR_UNEXPECTED_PARSER_EXCEPTION.format(exception=exc)

        return parsed_xml, logs
