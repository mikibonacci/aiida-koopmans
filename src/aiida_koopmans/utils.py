import shutil
import pathlib
import tempfile

import numpy as np
import functools

from aiida.common.exceptions import NotExistent
from aiida.orm import Code, Computer
from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.calculations.projwfc import ProjwfcCalculation
from aiida_wannier90.calculations.wannier90 import Wannier90Calculation

from ase import Atoms
from ase_koopmans import Atoms as AtomsKoopmans
from ase_koopmans import io
from ase_koopmans.io.espresso import kch_keys, kcp_keys, kcs_keys, pw_keys, w2kcw_keys

from aiida_koopmans.calculations.kcw import KcwCalculation
from aiida_koopmans.data.utils import generate_singlefiledata, generate_alpha_singlefiledata, produce_wannier90_files

LOCALHOST_NAME = "localhost-test"
KCW_BLOCKED_KEYWORDS = [t[1] for t in KcwCalculation._blocked_keywords]
PW_BLOCKED_KEYWORDS = [t[1] for t in PwCalculation._blocked_keywords]
PROJWFC_BLOCKED_KEYWORDS = [t[1] for t in ProjwfcCalculation._blocked_keywords]
WANNIER90_BLOCKED_KEYWORDS = [t[1] for t in Wannier90Calculation._BLOCKED_PARAMETER_KEYS]
ALL_BLOCKED_KEYWORDS = KCW_BLOCKED_KEYWORDS + PW_BLOCKED_KEYWORDS + WANNIER90_BLOCKED_KEYWORDS + PROJWFC_BLOCKED_KEYWORDS + [f'celldm({i})' for i in range (1,7)]

def get_builder_from_ase(calculator, step_data=None):
    return mapping_calculators[calculator.ext_out](calculator, step_data)

# Pw calculator.
def get_PwBaseWorkChain_from_ase(pw_calculator, step_data=None):
    from aiida import load_profile, orm
    from aiida_quantumespresso.common.types import ElectronicType
    from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain, PwCalculation

    load_profile()

    aiida_inputs = step_data.configuration
    calc_params = pw_calculator._parameters

    structure = None
    parent_folder = None
    for step_uid, val in step_data.steps.items():
        if "-scf" in step_uid and ("nscf" in pw_calculator.uid or "bands" in pw_calculator.uid):
            scf = orm.load_node(val["workchain"])
            structure = scf.inputs.pw.structure
            parent_folder = scf.outputs.remote_folder
    
    if not structure:
        if isinstance(pw_calculator.atoms, AtomsKoopmans):
            ase_atoms = Atoms.fromdict(pw_calculator.atoms.todict())
        structure = orm.StructureData(ase=ase_atoms) 

    pw_overrides = {
        "CONTROL": {},
        "SYSTEM": {"nosym": True, "noinv": True},
        "ELECTRONS": {},
    }

    for k in pw_keys['control']:
        if k in calc_params.keys() and k not in ALL_BLOCKED_KEYWORDS:
            pw_overrides["CONTROL"][k] = calc_params[k]

    for k in pw_keys['system']:
        if k in calc_params.keys() and k not in [ALL_BLOCKED_KEYWORDS]:
            pw_overrides["SYSTEM"][k] = calc_params[k]
        
    # I need to do the following otherwise the ecutrho can be set to different values,
    # and the wann2kc will fail because of this - there will be a FFT grid mismatch.
    if "ecutwfc" in pw_overrides["SYSTEM"].keys():
        pw_overrides["SYSTEM"]["ecutrho"] = pw_overrides["SYSTEM"]["ecutwfc"] * 4

    for k in pw_keys['electrons']:
        if k in calc_params.keys() and k not in ALL_BLOCKED_KEYWORDS:
            pw_overrides["ELECTRONS"][k] = calc_params[k]
    
    builder = PwBaseWorkChain.get_builder_from_protocol(
        code=aiida_inputs["pw_code"],
        structure=structure,
        overrides={
            "pseudo_family": step_data.pseudo_family, # TODO: automatic store of pseudos from koopmans folder, if not.
            "pw": {"parameters": pw_overrides},
        },
        electronic_type=ElectronicType.INSULATOR,
    )
    builder.pw.metadata = aiida_inputs["metadata"]
    if "npools" in aiida_inputs.keys(): builder.pw.parallelization = orm.Dict(dict={"npool": aiida_inputs["npools"]})
    builder.kpoints = orm.KpointsData()

    if pw_overrides["CONTROL"]["calculation"] in ["scf", "nscf"]:
        builder.kpoints.set_kpoints_mesh(calc_params["kpts"])
    elif pw_overrides["CONTROL"]["calculation"] == "bands":
        # here we need explicit kpoints
        builder.kpoints.set_kpoints(calc_params["kpts"].kpts,cartesian=False) # TODO: check cartesian false is correct.

    if parent_folder:
        builder.pw.parent_folder = parent_folder
    
    return builder, step_data

def get_Wannier90BandsWorkChain_builder_from_ase(w90_calculator, step_data=None):
    # get the builder from WannierizeWorkflow, but after we already initialized a Wannier90Calculator.
    # in this way we have everything we need for each different block of the wannierization step.

    from aiida import load_profile, orm
    from aiida_wannier90_workflows.common.types import WannierProjectionType
    from aiida_wannier90_workflows.utils.kpoints import get_explicit_kpoints_from_mesh
    from aiida_wannier90_workflows.utils.workflows.builder.serializer import (
        print_builder,
    )
    from aiida_wannier90_workflows.utils.workflows.builder.setter import (
        set_kpoints,
        set_num_bands,
        set_parallelization,
    )
    from aiida_wannier90_workflows.utils.workflows.builder.submit import (
        submit_and_add_group,
    )
    from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain, Wannier90WorkChain
    load_profile()

    #nscf = w90_calculator.parent_folder.creator.caller # PwBaseWorkChain
    nscf = None
    for step, val in step_data.steps.items():
            if "nscf" in str(step):
                nscf = orm.load_node(val["workchain"])
    if not nscf:
        raise ValueError("No nscf step found.")


    aiida_inputs = step_data.configuration

    codes = {
        "pw": aiida_inputs["pw_code"],
        "pw2wannier90": aiida_inputs["pw2wannier90_code"],
        "projwfc": aiida_inputs["projwfc_code"],
        "wannier90": aiida_inputs["wannier90_code"],
    }

    if all([p for p in nscf.inputs.pw.structure.pbc]):
        w90_wchain = Wannier90BandsWorkChain
    else:
        w90_wchain = Wannier90WorkChain
    
    # Use nscf explicit kpoints
    ## we do it first because if not a 3D system, kpath will except.
    kpoints = orm.KpointsData()
    kpoints.set_cell_from_structure(nscf.inputs.pw.structure)
    kpoints.set_kpoints(nscf.outputs.output_band.get_array('kpoints'), cartesian=False)
    
    builder = w90_wchain.get_builder_from_protocol(
            codes=codes,
            structure=nscf.inputs.pw.structure,
            pseudo_family=step_data.pseudo_family,
            protocol="moderate",
            projection_type=WannierProjectionType.ANALYTIC,
            print_summary=False,
            #bands_kpoints=kpoints
        )
    
    builder.wannier90.wannier90.kpoints = kpoints

    
    if all([p for p in nscf.inputs.pw.structure.pbc]):
        # set kpath using the WannierizeWFL data.
        kpoints_path = orm.KpointsData()
        
        k_coords = []
        k_labels = []

        k_path=w90_calculator.parameters.kpoint_path.kpts
        special_k = w90_calculator.parameters.kpoint_path.todict()["special_points"]
        k_linear,special_k_coords,special_k_labels = w90_calculator.parameters.kpoint_path.get_linear_kpoint_axis()
        t=0
        for coords,label in list(zip(special_k_coords,special_k_labels)):
            t = np.where(k_linear==coords)[0]
            k_labels.append([t[0],label])
            k_coords.append(special_k[label].tolist())
        
        kpoints_path.set_kpoints(k_path,labels=k_labels,cartesian=False)
        #del builder.bands_kpoints
        builder.kpoint_path  =  kpoints_path
    # else:
    #     k_path = kpoints.get_kpoints()
    #     k_labels = [[0,"G"]]
        
    


    # Start parameters and projections setting using the Wannier90Calculator data.
    params = builder.wannier90.wannier90.parameters.get_dict()

    del builder.scf
    del builder.nscf
    del builder.projwfc

    # pop dis_froz_max
    params.pop('dis_froz_max',None)
    
    for k,v in w90_calculator.parameters.items():
        if k not in ["kpoints","kpoint_path","projections"]:
            params[k] = v

    # projections in wannier90 format:
    converted_projs = []
    for proj in w90_calculator.todict()['_parameters']["projections"]:
        # for now we support only the following conversion:
        # proj={'fsite': [0.0, 0.0, 0.0], 'ang_mtm': 'sp3'} ==> converted_proj="f=0.0,0.0,0.0:sp3"
        if "fsite" in proj.keys():
            position = "f="+str(proj["fsite"]).replace("[","").replace("]","").replace(" ","")
        elif "site" in proj.keys():
            position = str(proj["site"])
        orbital = proj["ang_mtm"]
        converted_proj = position+":"+orbital
        converted_projs.append(converted_proj)

    builder.wannier90.wannier90.projections = orm.List(list=converted_projs)
    params.pop('auto_projections', None) # Uncomment this if you want analytic atomic projections

    ## END explicit atomic projections:

    # putting the fermi energy to make it work.
    try:
        fermi_energy = nscf.outputs.output_parameters.get_dict()["fermi_energy_up"]
    except:
        fermi_energy = nscf.outputs.output_parameters.get_dict()["fermi_energy"]
    params["fermi_energy"] = fermi_energy

    params = orm.Dict(dict=params)
    builder.wannier90.wannier90.parameters = params

    #resources
    builder.pw2wannier90.pw2wannier90.metadata = aiida_inputs.get("metadata_pw2wannier90", aiida_inputs["metadata"])

    default_w90_metadata_options_resources = {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1,
                "num_cores_per_mpiproc": 1
            }
    builder.wannier90.wannier90.metadata = aiida_inputs["metadata"]
    builder.wannier90.wannier90.metadata.options.resources = default_w90_metadata_options_resources

    builder.pw2wannier90.pw2wannier90.parent_folder = nscf.outputs.remote_folder

    # for now try this, as the get_fermi_energy_from_nscf + get_homo_lumo does not work for fixed occ.
    # maybe add some parsing (for fixed occ) in the aiida-wannier90-workflows/src/aiida_wannier90_workflows/utils/workflows/pw.py
    builder.wannier90.shift_energy_windows = False

    # adding pw2wannier90 parameters, required here. We should do in overrides.
    params_pw2wannier90 = builder.pw2wannier90.pw2wannier90.parameters.get_dict()
    params_pw2wannier90['inputpp']["wan_mode"] =  "standalone"
    
    if nscf.inputs.pw.parameters.get_dict()["SYSTEM"]["nspin"]>1: 
        params_pw2wannier90['inputpp']["spin_component"] = builder.wannier90.wannier90.parameters.get_dict()["spin"]
    builder.pw2wannier90.pw2wannier90.parameters = orm.Dict(dict=params_pw2wannier90)

    return builder, step_data


def get_projwfc_builder_from_ase(projwfc_calculator, step_data=None):
    from aiida import load_profile, orm
    from aiida_quantumespresso.calculations.projwfc import ProjwfcCalculation

    load_profile()

    """
    Convert a `ProjwfcCalculator` into an AiiDA `ProjwfcCalculation
    """

    aiida_inputs = step_data.configuration
    calc_params = projwfc_calculator._parameters

    # TODO: This is not needed, if we can just pass `orm.Dict(calc_params)` to the builder
    from koopmans.settings import ProjwfcSettingsDict

    projwfc_parameters = {}
    projwfcsettingsdict = ProjwfcSettingsDict()
    projwfc_keys = (
        projwfcsettingsdict.valid
        + list(projwfcsettingsdict.defaults.keys())
        + projwfcsettingsdict.are_paths
    )
    for k in projwfc_keys:
        if k in calc_params.keys() and k not in ALL_BLOCKED_KEYWORDS:
            projwfc_parameters[k] = calc_params[k]

    projwfc_parameters['filpdos'] = 'aiida'

    builder = ProjwfcCalculation.get_builder()
    builder.code = orm.load_code(aiida_inputs["projwfc_code"])
    builder.parameters = orm.Dict({"PROJWFC": projwfc_parameters})
    builder.metadata = aiida_inputs.get("metadata_projwfc", aiida_inputs["metadata"])
    builder.metadata.options.additional_retrieve_list = ['aiida.pdos*']

    parent_calculators = [
        f[0].parent_process.uid for f in projwfc_calculator.linked_files.values() if f[0] is not None
    ]

    if len(set(parent_calculators)) > 1:
        raise ValueError("More than one parent calculator found.")
    elif len(set(parent_calculators)) == 1:
        if "remote_folder" in step_data.steps[parent_calculators[0]]:
            builder.parent_folder = orm.load_node(
                step_data.steps[parent_calculators[0]]["remote_folder"]
            )

    return builder, step_data

def get_kcw_builder_from_ase(kcw_calculator, step_data=None):

    from aiida import load_profile, orm
    load_profile()
    
    aiida_inputs = step_data.configuration
    
    # here we should find the parent folder and the wann files, merged or not (single block for emp or occ manifold).
    parent_folder = None
    wann_u_mat = None
    wann_emp_u_mat = None
    wann_emp_u_dis_mat = None
    wann_centres_xyz = None
    wann_emp_centres_xyz = None
    alpha = None
    read_unitary_matrix = False
    kcw_at_ks = True
    spin = None
    for spin_c in ["spin_1","spin_2"]:
        if spin_c in kcw_calculator.uid:
            spin = spin_c
        
    for step_uid, val in step_data.steps.items():
        if "wannier90" in step_uid:
            read_unitary_matrix = True
            kcw_at_ks = False
        if "-dft" in step_uid:
            dft = orm.load_node(val["workchain"])
            parent_folder = dft.outputs.remote_folder
        if "nscf" in step_uid:
            nscf = orm.load_node(val["workchain"])
            parent_folder = nscf.outputs.remote_folder
        if "kcw_wannier" in step_uid and "workchain" in val:
            # here we need to distinguish between the two spin channels, if present,
            # and only proceed if we are in the same spin channel.
            if "spin" in kcw_calculator.uid:
                for spin_channel in ["spin_1", "spin_2"]:
                    if spin_channel in step_uid and spin_channel in kcw_calculator.uid:
                        w2kc = orm.load_node(val["workchain"])
                        parent_folder = w2kc.outputs.remote_folder
                    else: # not really spin channel, so we just provide the same parent folder.
                        w2kc = orm.load_node(val["workchain"])
                        parent_folder = w2kc.outputs.remote_folder
            else:
                w2kc = orm.load_node(val["workchain"])
                parent_folder = w2kc.outputs.remote_folder
                    
        
        # alphas singlefiledata files. we take the kcw_ham step as actually the alphas are stored there.
        if "kcw_ham" in step_uid:
            if "spin" in kcw_calculator.uid:
                if spin not in step_uid: 
                    continue
            kcw_calculator.write_alphas()
            if "file_alpharef.txt" in val.get('input_files',{}):
                alpha = orm.load_node(val['input_files']['file_alpharef.txt'])
        
        # Wannier90 SinglefileData merged files:
        # TODO: spin channel distinction.
        if "merge_occ_wannier_u" in step_uid:
            wann_u_mat = orm.load_node(val['input_files']['wannier90_u.mat'])
        if "merge_occ_wannier_centers" in step_uid:
            wann_centres_xyz = orm.load_node(val['input_files']['wannier90_centres.xyz'])
        if "merge_emp_wannier_u" in step_uid: # TODO: check if this is correct
            wann_emp_u_mat = orm.load_node(val['input_files']['wannier90_u.mat'])
        if "merge_emp_wannier_centers" in step_uid:
            wann_emp_centres_xyz = orm.load_node(val['input_files']['wannier90_centres.xyz'])
        if "merge_emp_wannier_u_dis" in step_uid:
            wann_emp_u_dis_mat = orm.load_node(val['input_files']['wannier90_u_dis.mat'])
        
        
        
    # RemoteData folders: this is when only one block in occ or emp manifold.
    # Instead of the SinglefileData (as searched above), we have only the RemoteData 
    # of the wannnier90 calc. linking this, will copy all the needed wann files (u, centres, etc.)
    # TODO: explain this logic.
    tmp_wann_emp_u_mat = None

    for step_uid, val in step_data.steps.items():
        # the first hit is the single block of occ manifold,
        # so we assign it and then we never hit again this block.
        if not wann_u_mat and "03-wannier90" in step_uid:
            if "spin" in kcw_calculator.uid:
                spin_wannier = get_spin_wannier_wkchain(orm.load_node(val["workchain"]))
                if spin_wannier != spin:
                    continue
            wann_u_mat = orm.load_node(val["remote_folder"])

        # we continue updating it up to the last hit.
        # the last hit is the single block of emp manifold
        if not wann_emp_u_mat and "03-wannier90" in step_uid: 
            if "spin" in kcw_calculator.uid:
                spin_wannier = get_spin_wannier_wkchain(orm.load_node(val["workchain"]))
                if spin_wannier != spin:
                    continue
            tmp_wann_emp_u_mat = orm.load_node(val["remote_folder"])
        
    if tmp_wann_emp_u_mat: wann_emp_u_mat = tmp_wann_emp_u_mat    
        
    # get the kcw calculator ext_out: we have three cases: w2ko, kso, kho
    ext_out = kcw_calculator.ext_out
    
    control_namelist = kcw_inputs_keys[ext_out]['control']
    wannier_namelist = kcw_inputs_keys[ext_out]['wannier']

    control_dict = {
        k: v if k in control_namelist else None
        for k, v in kcw_calculator.parameters.items()
        if k not in ALL_BLOCKED_KEYWORDS
    }

    control_dict["calculation"] = "wann2kcw"
    for k in list(control_dict):
        if control_dict[k] is None:
            control_dict.pop(k)
    if read_unitary_matrix:
        control_dict["read_unitary_matrix"] = read_unitary_matrix
        control_dict["kcw_at_ks"] = kcw_at_ks

    wannier_dict = {
        k: v if k in wannier_namelist else None
        for k, v in kcw_calculator.parameters.items()
        # ? Using all here, as blocked Wannier90 keywords doesn't contain 'seedname', but kcw does
        if k not in ALL_BLOCKED_KEYWORDS
    }
    for k in list(wannier_dict):
        if wannier_dict[k] is None:
            wannier_dict.pop(k)

    screening_dict = {
        k: v if k in kcs_keys['screen'] else None
        for k, v in kcw_calculator.parameters.items()
        if k not in ALL_BLOCKED_KEYWORDS
    }
    for k in list(screening_dict):
        if screening_dict[k] is None:
            screening_dict.pop(k)
    
    ham_dict = {
        k: v if k in kch_keys['ham'] else None
        for k, v in kcw_calculator.parameters.items()
        if k not in ALL_BLOCKED_KEYWORDS
    }
    for k in list(ham_dict):
        if ham_dict[k] is None:
            ham_dict.pop(k)
    
    # NOTE, TODO: to be deleted! but I cannot do it correctly otherwise.
    # if too many nbnd, I need to set it by hands for now. 
    # otherwise it will do num_wann_emp = nbnd - num_wann_occ, but this should depend on the wannier projections!!!
    #hard_coded = 66 if spin == "spin_1" else 66
    #wannier_dict["num_wann_emp"] = hard_coded
    
    kcw_params = {
        "CONTROL": control_dict,
        "WANNIER": wannier_dict,
    }
    if ext_out == ".kso":
        kcw_params["SCREEN"] = screening_dict
        kcw_params["CONTROL"]["calculation"] = "screen"
    elif ext_out == ".kho":
        kcw_params["CONTROL"]["calculation"] = "ham"
        kcw_params["HAM"] = ham_dict

    # builder. 
    builder = KcwCalculation.get_builder()
    builder.parameters = orm.Dict(kcw_params)
    builder.code = orm.load_code(aiida_inputs["kcw_code"])
    
    builder.metadata = aiida_inputs["metadata"]
    if "metadata_kcw" in aiida_inputs:
        builder.metadata = aiida_inputs["metadata_kcw"]
        
    if ext_out == ".kho":
        # I provide kpoints as an array (output in the wannierized band structure), so I need to convert them. 
        kpoints = orm.KpointsData()
        if len(kcw_calculator._parameters.kpts.kpts) == 0:
            kpts = np.array([[0.0,0.0,0.0]])
        else:
            kpts = kcw_calculator._parameters.kpts.kpts
        kpoints.set_kpoints(kpts, cartesian=False)
        builder.kpoints = kpoints 
    
    builder.parent_folder = parent_folder

    if control_dict.get(
        "read_unitary_matrix", read_unitary_matrix
    ):
        if wann_u_mat: builder.wann_u_mat = wann_u_mat
        if wann_emp_u_mat: builder.wann_emp_u_mat = wann_emp_u_mat
        if wann_emp_u_dis_mat: builder.wann_emp_u_dis_mat = wann_emp_u_dis_mat
        if wann_centres_xyz: builder.wann_centres_xyz = wann_centres_xyz
        if wann_emp_centres_xyz: builder.wann_emp_centres_xyz = wann_emp_centres_xyz
        
    if alpha:
        builder.alpha = alpha
            
    return builder, step_data

def get_spin_wannier_wkchain(node):
    spin = node.inputs.wannier90.wannier90.parameters.get_dict()["spin"]
    return "spin_1" if spin == "up" else "spin_2"

## Here we have the mapping for the calculators initialization. used in the `aiida_calculate_trigger`.
mapping_calculators = {
    ".pwo" : get_PwBaseWorkChain_from_ase,
    ".wout": get_Wannier90BandsWorkChain_builder_from_ase,
    ".pro": get_projwfc_builder_from_ase,
    ".w2ko": get_kcw_builder_from_ase,
    ".kso": get_kcw_builder_from_ase,
    ".kho": get_kcw_builder_from_ase,
}

kcw_inputs_keys = {
    ".w2ko": w2kcw_keys,
    ".kso": kcs_keys,
    ".kho": kch_keys,
}

# read the output file, mimicking the read_results method of ase-koopmans: https://github.com/elinscott/ase_koopmans/blob/master/ase/calculators/espresso/_espresso.py
def read_output_file(calculator, retrieved, inner_remote_folder=None):
    """
    Read the output file of a calculator using ASE io.read() method but parsing the AiiDA outputs.
    NB: calculator (ASE) should contain the related AiiDA workchain as attribute.
    """
    # if inner_remote_folder:
    #    retrieved = inner_remote_folder
    # else:
    # retrieved = workchain.outputs.retrieved
    with tempfile.TemporaryDirectory() as dirpath:
        # Open the output file from the AiiDA storage and copy content to the temporary file
        for filename in retrieved.base.repository.list_object_names():
            if '.out' in filename or '.wout' in filename:
                # Create the file with the desired name
                readable_filename = calculator.label.split("/")[-1]+calculator.ext_out
                temp_file = pathlib.Path(dirpath) / readable_filename
                with retrieved.open(filename, 'rb') as handle:
                    temp_file.write_bytes(handle.read())
                output = io.read(temp_file)
    return output


def dump_pdos_outputs(calculator, retrieved):
    """
    Dump the `pdos` output files of a projwfc.x calculation run via AiiDA to a temporary directory which is returned.
    """

    output_dir = pathlib.Path('/tmp') / pathlib.Path(tempfile.mkdtemp()).parts[-1]
    output_dir.mkdir(exist_ok=True, parents=True)

    for filename in retrieved.base.repository.list_object_names():
        if ".pdos" in filename:
            # Create the file with the desired name
            output_file = pathlib.Path(output_dir) / (
                filename.replace("aiida", f"{calculator.parameters.filpdos}")
            )
            with retrieved.open(filename, "rb") as handle:
                output_file.write_bytes(handle.read())

    return output_dir


def delete_directory(dir_path):
    dir_path = pathlib.Path(dir_path)
    for child in dir_path.iterdir():
        if child.is_dir():
            delete_directory(child)
        else:
            child.unlink()
    dir_path.rmdir()
