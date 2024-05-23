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

def produce_wannier90_files(calc_w90,merge_directory_name,method="dfpt"):
    """producing the wannier90 files in the case of just one occ and/or one emp blocks.

    Args:
        wannierize_workflow (WannierizeWorkflow): WannierizeWorkflow which is doing the splitted wannierization. 
        merge_directory_name (str): "occ" or "emp", as obtained in the WannierizeWorkflow
    
    Returns:
        dict: dictionary containing SingleFileData of the files: hr, u and centres for occ and emp, then u_dis if dfpt.
    """
    hr_file = calc_w90.wchain.outputs.wannier90.retrieved.get_object_content('aiida' + '_hr.dat')
    u_file = calc_w90.wchain.outputs.wannier90.retrieved.get_object_content('aiida' + '_u.mat')
    centres_file = calc_w90.wchain.outputs.wannier90.retrieved.get_object_content('aiida' + '_centres.xyz')
    
    hr_singlefile = generate_singlefiledata('aiida' + '_hr.dat', hr_file)
    u_singlefile = generate_singlefiledata('aiida' + '_u.mat', u_file)
    centres_singlefile = generate_singlefiledata('aiida' + '_centres.xyz', centres_file)
    
    standard_dictionary =  {'hr_dat':hr_singlefile, "u_mat": u_singlefile, "centres_xyz": centres_singlefile}
    
    if method == 'dfpt' and merge_directory_name == "emp":
        u_dis_file = calc_w90.wchain.outputs.wannier90.retrieved.get_object_content('aiida' + '_u_dis.mat')
        u_dis_singlefile = generate_singlefiledata('aiida' + '_u_dis.mat', u_dis_file)
        standard_dictionary["u_dis_mat"] = u_dis_singlefile
        
    return standard_dictionary

def generate_alpha_singlefiledata(calc):
    # self.alphas is a list of alpha values indexed by spin index and then band index. Meanwhile, kcw.x takes a
    # single file for the alphas (rather than splitting between filled/empty) and does not have two columns for
    # spin up then spin down
    assert calc.alphas is not None, 'You have not provided screening parameters to this calculator'
    if not len(calc.alphas) == 1:
        raise NotImplementedError('KoopmansHamCalculator yet to be implemented for spin-polarized systems')
    calc.alphas_files = {}
    [alphas] = calc.alphas
    filling = [True for _ in range(len(alphas))]
    
    a_filled = [a for a, f in zip(alphas, filling) if f]
    a_empty = [a for a, f in zip(alphas, filling) if not f]
    
    with tempfile.TemporaryDirectory() as dirpath:
        # Open the output file from the AiiDA storage and copy content to the temporary file
        a_filled = [a for a, f in zip(alphas, filling) if f]
        a_empty = [a for a, f in zip(alphas, filling) if not f]
        for alphas, suffix in zip([a_filled, a_empty], ['', '_empty']):
            temp_file = pathlib.Path(dirpath) / f'file_alpharef{suffix}.txt'
            with open(temp_file, 'w') as fd:
                fd.write('{}\n'.format(len(alphas)))
                for i, a in enumerate(alphas):
                    fd.write('{} {} 1.0\n'.format(i + 1, a))
                    
                calc.alphas_files["alpha"+suffix] = SingleFileData(temp_file)
                