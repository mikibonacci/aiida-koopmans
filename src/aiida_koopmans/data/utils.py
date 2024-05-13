import pathlib
import tempfile
from aiida.plugins import DataFactory
SingleFileData = DataFactory('core.singlefile')

def generate_singlefiledata(filename, flines):
    with tempfile.TemporaryDirectory() as dirpath:
        # Open the output file from the AiiDA storage and copy content to the temporary file
        temp_file = pathlib.Path(dirpath) / filename
        with open(temp_file, 'w') as fd:
            fd.write(''.join(flines))
        
        file = SingleFileData(temp_file)

        return file

def produce_wannier90_files(wannierize_workflow,merge_directory_name):
    """producing the wannier90 files in the case of just one occ and/or one emp blocks.

    Args:
        wannierize_workflow (WannierizeWorkflow): WannierizeWorkflow which is doing the splitted wannierization. 
        merge_directory_name (str): "occ" or "emp", as obtained in the WannierizeWorkflow
    
    Returns:
        dict: dictionary containing SingleFileData of the files: hr, u and centres for occ and emp, then u_dis if dfpt.
    """
    hr_file = wannierize_workflow.w90_wchains[merge_directory_name][0].outputs.wannier90.retrieved.get_object_content('aiida' + '_hr.dat')
    u_file = wannierize_workflow.w90_wchains[merge_directory_name][0].outputs.wannier90.retrieved.get_object_content('aiida' + '_u.mat')
    centres_file = wannierize_workflow.w90_wchains[merge_directory_name][0].outputs.wannier90.retrieved.get_object_content('aiida' + '_centres.xyz')
    
    hr_singlefile = generate_singlefiledata('aiida' + '_hr.dat', hr_file)
    u_singlefile = generate_singlefiledata('aiida' + '_u.mat', u_file)
    centres_singlefile = generate_singlefiledata('aiida' + '_centres.xyz', centres_file)
    
    standard_dictionary =  {'hr_dat':hr_singlefile, "u_mat": u_singlefile, "centres_xyz": centres_singlefile}
    
    if wannierize_workflow.parameters.method == 'dfpt' and merge_directory_name == "emp":
        u_dis_file = wannierize_workflow.w90_wchains[merge_directory_name][0].outputs.wannier90.retrieved.get_object_content('aiida' + '_u_dis.mat')
        u_dis_singlefile = generate_singlefiledata('aiida' + '_u_dis.mat', u_dis_file)
        standard_dictionary["u_dis_mat"] = u_dis_singlefile
        
    return standard_dictionary