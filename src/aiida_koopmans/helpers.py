""" Helper functions for automatically setting up computer & code.
Helper functions for setting up

 1. An AiiDA localhost computer
 2. A "diff" code on localhost

Note: Point 2 is made possible by the fact that the ``diff`` executable is
available in the PATH on almost any UNIX system.
"""

import shutil
import pathlib
import tempfile

import numpy as np
import functools

from aiida.common.exceptions import NotExistent
from aiida.orm import Code, Computer
from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_wannier90.calculations.wannier90 import Wannier90Calculation
from ase import io
from ase.io.espresso import kch_keys, kcp_keys, kcs_keys, pw_keys, w2kcw_keys

from aiida_koopmans.calculations.kcw import KcwCalculation
from aiida_koopmans.data.utils import generate_singlefiledata, generate_alpha_singlefiledata

"""
ASE calculator MUST have `wchain` attribute (the related AiiDA WorkChain) to be able to use these functions!
"""

LOCALHOST_NAME = "localhost-test"
KCW_BLOCKED_KEYWORDS = [t[1] for t in KcwCalculation._blocked_keywords]
PW_BLOCKED_KEYWORDS = [t[1] for t in PwCalculation._blocked_keywords]
WANNIER90_BLOCKED_KEYWORDS = [t[1] for t in Wannier90Calculation._BLOCKED_PARAMETER_KEYS]
ALL_BLOCKED_KEYWORDS = KCW_BLOCKED_KEYWORDS + PW_BLOCKED_KEYWORDS + WANNIER90_BLOCKED_KEYWORDS + [f'celldm({i})' for i in range (1,7)]


executables = {
    "koopmans": "diff",
}


def get_path_to_executable(executable):
    """Get path to local executable.
    :param executable: Name of executable in the $PATH variable
    :type executable: str
    :return: path to executable
    :rtype: str
    """
    path = shutil.which(executable)
    if path is None:
        raise ValueError(f"'{executable}' executable not found in PATH.")
    return path


def get_computer(name=LOCALHOST_NAME, workdir=None):
    """Get AiiDA computer.
    Loads computer 'name' from the database, if exists.
    Sets up local computer 'name', if it isn't found in the DB.

    :param name: Name of computer to load or set up.
    :param workdir: path to work directory
        Used only when creating a new computer.
    :return: The computer node
    :rtype: :py:class:`aiida.orm.computers.Computer`
    """

    try:
        computer = Computer.objects.get(label=name)
    except NotExistent:
        if workdir is None:
            workdir = tempfile.mkdtemp()

        computer = Computer(
            label=name,
            description="localhost computer set up by aiida_diff tests",
            hostname=name,
            workdir=workdir,
            transport_type="core.local",
            scheduler_type="core.direct",
        )
        computer.store()
        computer.set_minimum_job_poll_interval(0.0)
        computer.configure()

    return computer

def get_code(entry_point, computer):
    """Get local code.
    Sets up code for given entry point on given computer.

    :param entry_point: Entry point of calculation plugin
    :param computer: (local) AiiDA computer
    :return: The code node
    :rtype: :py:class:`aiida.orm.nodes.data.code.installed.InstalledCode`
    """

    try:
        executable = executables[entry_point]
    except KeyError as exc:
        raise KeyError(
            f"Entry point '{entry_point}' not recognized. Allowed values: {list(executables.keys())}"
        ) from exc

    codes = Code.objects.find(  # pylint: disable=no-member
        filters={"label": executable}
    )
    if codes:
        return codes[0]

    path = get_path_to_executable(executable)
    code = Code(
        input_plugin_name=entry_point,
        remote_computer_exec=[computer, path],
    )
    code.label = executable
    return code.store()

# read the output file, mimicking the read_results method of ase-koopmans: https://github.com/elinscott/ase_koopmans/blob/master/ase/calculators/espresso/_espresso.py
def read_output_file(calculator, inner_remote_folder=None):
    """
    Read the output file of a calculator using ASE io.read() method but parsing the AiiDA outputs. 
    NB: calculator (ASE) should contain the related AiiDA workchain as attribute.
    """
    if inner_remote_folder:
        retrieved = inner_remote_folder
    else:
        retrieved = calculator.wchain.outputs.retrieved
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

def get_output_content(calculator, filename, mode="r", inner_remote_folder=None):
    """
    This is needed for parsing AiiDA stored files for further manipulation. E.g., merge wannier files.
    NB: calculator (ASE) should contain the related AiiDA workchain as attribute.
    """
    if inner_remote_folder:
        retrieved = inner_remote_folder
    else:
        retrieved = calculator.wchain.outputs.retrieved
    content = retrieved.get_object_content(filename,mode=mode)
    if mode=="rb": return content
    content = content.split("\n")
    for line in range(len(content)):
        content[line] += "\n"
    return content[:-1] # this is the analogous of the file.readlines()
    
# Pw calculator.
def get_builder_from_ase(pw_calculator):
    from aiida import load_profile, orm
    from aiida_quantumespresso.common.types import ElectronicType
    from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain, PwCalculation

    load_profile()

    """
    We should check automatically on the accepted keywords in PwCalculation and where are. Should be possible.
    we suppose that the calculator has an attribute called mode e.g.

    pw_calculator.parameters.mode = {
        "pw_code": "pw-7.2-ok@localhost",
        "metadata": {
        "options": {
            "max_wallclock_seconds": 3600,
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1,
                "num_cores_per_mpiproc": 1
            },
            "custom_scheduler_commands": "export OMP_NUM_THREADS=1"
        }
    }
    }
    """
    aiida_inputs = pw_calculator.parameters.mode
    calc_params = pw_calculator._parameters
    structure = orm.StructureData(ase=pw_calculator.atoms)

    pw_overrides = {
        "CONTROL": {},
        "SYSTEM": {"nosym": True, "noinv": True},
        "ELECTRONS": {},
    }

    for k in pw_keys['control']:
        if k in calc_params.keys() and k not in ALL_BLOCKED_KEYWORDS:
            pw_overrides["CONTROL"][k] = calc_params[k]

    for k in pw_keys['system']:
        if k in calc_params.keys() and k not in ALL_BLOCKED_KEYWORDS:
            pw_overrides["SYSTEM"][k] = calc_params[k]

    for k in pw_keys['electrons']:
        if k in calc_params.keys() and k not in ALL_BLOCKED_KEYWORDS:
            pw_overrides["ELECTRONS"][k] = calc_params[k]

    builder = PwBaseWorkChain.get_builder_from_protocol(
        code=aiida_inputs["pw_code"],
        structure=structure,
        overrides={
            "pseudo_family": "PseudoDojo/0.4/PBE/SR/standard/upf",
            "pw": {"parameters": pw_overrides},
        },
        electronic_type=ElectronicType.INSULATOR,
    )
    builder.pw.metadata = aiida_inputs["metadata"]

    builder.kpoints = orm.KpointsData()
    builder.kpoints.set_kpoints_mesh(calc_params["kpts"])

    if hasattr(pw_calculator, "parent_folder"):
        builder.pw.parent_folder = pw_calculator.parent_folder

    return builder

def from_wann2kc_to_KcwCalculation(wann2kc_calculator):
    """
    The input parent folder is meant to be set later, at least for now.
    """

    from aiida import load_profile, orm

    load_profile()

    builder = KcwCalculation.get_builder()
    wann2kc_control_namelist = w2kcw_keys['control']
    wann2kc_wannier_namelist = w2kcw_keys['wannier']


    control_dict = {
        k: v if k in wann2kc_control_namelist else None
        for k, v in wann2kc_calculator.parameters.items()
        if k not in ALL_BLOCKED_KEYWORDS
    }

    control_dict["calculation"] = "wann2kcw"

    for k in list(control_dict):
        if control_dict[k] is None:
            control_dict.pop(k)

    wannier_dict = {
        k: v if k in wann2kc_wannier_namelist else None
        for k, v in wann2kc_calculator.parameters.items()
        # ? Using all here, as blocked Wannier90 keywords doesn't contain 'seedname', but kcw does
        if k not in ALL_BLOCKED_KEYWORDS
    }

    for k in list(wannier_dict):
        if wannier_dict[k] is None:
            wannier_dict.pop(k)

    wann2kcw_params = {
        "CONTROL": control_dict,
        "WANNIER": wannier_dict,
    }

    builder.parameters = orm.Dict(wann2kcw_params)
    builder.code = orm.load_code(wann2kc_calculator.parameters.mode["kcw_code"])
    builder.metadata = wann2kc_calculator.parameters.mode["metadata"]
    if "metadata_kcw" in wann2kc_calculator.parameters.mode:
        builder.metadata = wann2kc_calculator.parameters.mode["metadata_kcw"]
    builder.parent_folder = wann2kc_calculator.parent_folder

    if hasattr(wann2kc_calculator, "w90_files"):
        builder.wann_u_mat = wann2kc_calculator.w90_files["occ"]["u_mat"]
        builder.wann_emp_u_mat = wann2kc_calculator.w90_files["emp"]["u_mat"]
        builder.wann_emp_u_dis_mat = wann2kc_calculator.w90_files["emp"][
            "u_dis_mat"
        ]
        builder.wann_centres_xyz = wann2kc_calculator.w90_files["occ"][
            "centres_xyz"
        ]
        builder.wann_emp_centres_xyz = wann2kc_calculator.w90_files["emp"][
            "centres_xyz"
        ]

    return builder

def from_kcwham_to_KcwCalculation(kcw_calculator):
    """
    The input parent folder is meant to be set later, at least for now.
    """

    from aiida import load_profile, orm

    from aiida_koopmans.calculations.kcw import KcwCalculation

    load_profile()

    builder = KcwCalculation.get_builder()

    kch_control_namelist = kch_keys['control']
    kch_wannier_namelist = kch_keys['wannier']
    kch_ham_namelist = kch_keys['ham']

    control_dict = {
        k: v if k in kch_control_namelist else None
        for k, v in kcw_calculator.parameters.items()
        if k not in ALL_BLOCKED_KEYWORDS
    }
    control_dict["calculation"] = "ham"

    for k in list(control_dict):
        if control_dict[k] is None:
            control_dict.pop(k)


    wannier_dict = {
        k: v if k in kch_wannier_namelist else None
        for k, v in kcw_calculator.parameters.items()
        if k not in ALL_BLOCKED_KEYWORDS
    }

    for k in list(wannier_dict):
        if wannier_dict[k] is None:
            wannier_dict.pop(k)

    ham_dict = {
        k: v if k in kch_ham_namelist else None
        for k, v in kcw_calculator.parameters.items()
        if k not in ALL_BLOCKED_KEYWORDS
    }

    for k in list(ham_dict):
        if ham_dict[k] is None:
            ham_dict.pop(k)
    
    # for now always true as we skip the smooth interpolation procedure.    
    if not any(kcw_calculator.atoms.pbc):
        ham_dict["do_bands"] = False
    else:
        ham_dict["do_bands"] = True

    kcw_ham_params = {
        "CONTROL": control_dict,
        "WANNIER": wannier_dict,
        "HAM": ham_dict,
    }

    builder.parameters = orm.Dict(kcw_ham_params)
    builder.code = orm.load_code(kcw_calculator.parameters.mode["kcw_code"])
    builder.metadata = kcw_calculator.parameters.mode["metadata"]
    if "metadata_kcw" in kcw_calculator.parameters.mode:
        builder.metadata = kcw_calculator.parameters.mode["metadata_kcw"]
    builder.parent_folder = kcw_calculator.parent_folder

    if hasattr(kcw_calculator, "w90_files") and control_dict.get(
        "read_unitary_matrix", False
    ):
        builder.wann_u_mat = kcw_calculator.w90_files["occ"]["u_mat"]
        builder.wann_emp_u_mat = kcw_calculator.w90_files["emp"]["u_mat"]
        builder.wann_emp_u_dis_mat = kcw_calculator.w90_files["emp"]["u_dis_mat"]
        builder.wann_centres_xyz = kcw_calculator.w90_files["occ"]["centres_xyz"]
        builder.wann_emp_centres_xyz = kcw_calculator.w90_files["emp"][
            "centres_xyz"
        ]
        
    if hasattr(kcw_calculator, "kpoints"):
        # I provide kpoints as an array (output in the wannierized band structure), so I need to convert them. 
        kpoints = orm.KpointsData()
        kpoints.set_kpoints(kcw_calculator.kpoints)
        builder.kpoints = kpoints
        
    if hasattr(kcw_calculator, "alphas_files"):
        builder.alpha_occ = kcw_calculator.alphas_files["alpha"]
        builder.alpha_emp = kcw_calculator.alphas_files["alpha_empty"]
    
    return builder

def from_kcwscreen_to_KcwCalculation(kcw_calculator):
    """
    The input parent folder is meant to be set later, at least for now.
    """

    from aiida import load_profile, orm

    from aiida_koopmans.calculations.kcw import KcwCalculation

    load_profile()

    builder = KcwCalculation.get_builder()

    kcs_control_namelist = kcs_keys['control']
    kcs_wannier_namelist = kcs_keys['wannier']
    kcs_screening_namelist = kcs_keys['screen']

    control_dict = {
        k: v if k in kcs_control_namelist else None
        for k, v in kcw_calculator.parameters.items()
        if k not in ALL_BLOCKED_KEYWORDS
    }
    control_dict["calculation"] = "screen"

    for k in list(control_dict):
        if control_dict[k] is None:
            control_dict.pop(k)

    wannier_dict = {
        k: v if k in kcs_wannier_namelist else None
        for k, v in kcw_calculator.parameters.items()
        if k not in ALL_BLOCKED_KEYWORDS
    }

    for k in list(wannier_dict):
        if wannier_dict[k] is None:
            wannier_dict.pop(k)

    screening_dict = {
        k: v if k in kcs_screening_namelist else None
        for k, v in kcw_calculator.parameters.items()
        if k not in ALL_BLOCKED_KEYWORDS
    }

    for k in list(screening_dict):
        if screening_dict[k] is None:
            screening_dict.pop(k)

    kcw_screen_params = {
        "CONTROL": control_dict,
        "WANNIER": wannier_dict,
        "SCREEN": screening_dict,
    }

    builder.parameters = orm.Dict(kcw_screen_params)
    builder.code = orm.load_code(kcw_calculator.parameters.mode["kcw_code"])
    builder.metadata = kcw_calculator.parameters.mode["metadata"]
    if "metadata_kcw" in kcw_calculator.parameters.mode:
        builder.metadata = kcw_calculator.parameters.mode["metadata_kcw"]
    builder.parent_folder = kcw_calculator.parent_folder

    if hasattr(kcw_calculator, "w90_files") and control_dict.get(
        "read_unitary_matrix", False
    ):
        builder.wann_u_mat = kcw_calculator.w90_files["occ"]["u_mat"]
        builder.wann_emp_u_mat = kcw_calculator.w90_files["emp"]["u_mat"]
        builder.wann_emp_u_dis_mat = kcw_calculator.w90_files["emp"]["u_dis_mat"]
        builder.wann_centres_xyz = kcw_calculator.w90_files["occ"]["centres_xyz"]
        builder.wann_emp_centres_xyz = kcw_calculator.w90_files["emp"][
            "centres_xyz"
        ]
    
    return builder

def get_wannier90bandsworkchain_builder_from_ase(w90_calculator):
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
    from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain
    load_profile()

    nscf = w90_calculator.parent_folder.creator.caller # PwBaseWorkChain
    aiida_inputs = w90_calculator.parameters.mode

    codes = {
        "pw": aiida_inputs["pw_code"],
        "pw2wannier90": aiida_inputs["pw2wannier90_code"],
        #"projwfc": aiida_inputs["projwfc_code"],
        "wannier90": aiida_inputs["wannier90_code"],
    }

    builder = Wannier90BandsWorkChain.get_builder_from_protocol(
            codes=codes,
            structure=nscf.inputs.pw.structure,
            pseudo_family="PseudoDojo/0.4/PBE/FR/standard/upf",
            protocol="fast",
            projection_type=WannierProjectionType.ANALYTIC,
            print_summary=False,
        )

    # Use nscf explicit kpoints
    kpoints = orm.KpointsData()
    kpoints.set_cell_from_structure(builder.structure)
    kpoints.set_kpoints(nscf.outputs.output_band.get_array('kpoints'),cartesian=False)
    builder.wannier90.wannier90.kpoints = kpoints

    # set kpath using the WannierizeWFL data.
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
    
    kpoints_path = orm.KpointsData()
    kpoints_path.set_kpoints(k_path,labels=k_labels,cartesian=False)
    builder.kpoint_path  =  kpoints_path


    # Start parameters and projections setting using the Wannier90Calculator data.
    params = builder.wannier90.wannier90.parameters.get_dict()

    del builder.scf
    del builder.nscf
    del builder.projwfc

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
    builder.pw2wannier90.pw2wannier90.metadata = aiida_inputs["metadata"]

    default_w90_metadata = {
          "options": {
            "max_wallclock_seconds": 3600,
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1,
                "num_cores_per_mpiproc": 1
            },
            "custom_scheduler_commands": "export OMP_NUM_THREADS=1"
        }
      }
    builder.wannier90.wannier90.metadata = aiida_inputs.get('metadata_w90', default_w90_metadata)

    builder.pw2wannier90.pw2wannier90.parent_folder = nscf.outputs.remote_folder

    # for now try this, as the get_fermi_energy_from_nscf + get_homo_lumo does not work for fixed occ.
    # maybe add some parsing (for fixed occ) in the aiida-wannier90-workflows/src/aiida_wannier90_workflows/utils/workflows/pw.py
    builder.wannier90.shift_energy_windows = False

    # adding pw2wannier90 parameters, required here. We should do in overrides.
    params_pw2wannier90 = builder.pw2wannier90.pw2wannier90.parameters.get_dict()
    params_pw2wannier90['inputpp']["wan_mode"] =  "standalone"
    if nscf.inputs.pw.parameters.get_dict()["SYSTEM"]["nspin"]>1: params_pw2wannier90['inputpp']["spin_component"] = "up"
    builder.pw2wannier90.pw2wannier90.parameters = orm.Dict(dict=params_pw2wannier90)


    return builder

# Decorators.

## Here we have the mapping for the calculators initialization. used in the `aiida_calculate_trigger`.
mapping_calculators = {
    ".pwo" : get_builder_from_ase,
    ".wout": get_wannier90bandsworkchain_builder_from_ase,
    ".w2ko": from_wann2kc_to_KcwCalculation,
    ".kso": from_kcwscreen_to_KcwCalculation,
    ".kho": from_kcwham_to_KcwCalculation,
}

## Calculate step
def aiida_pre_calculate_trigger(_pre_calculate):
    # This wraps the _pre_calculate method. 
    @functools.wraps(_pre_calculate)
    def wrapper_aiida_trigger(self):
        if self.parameters.mode == "ase":
            return _pre_calculate(self,)
        else:
            pass
    return wrapper_aiida_trigger

def aiida_calculate_trigger(_calculate):
    # This wraps the _calculate method. submits AiiDA if we are not in the "ase" mode, otherwise it behaves like in the standard Koopmans run.
    @functools.wraps(_calculate)
    def wrapper_aiida_trigger(self):
        if self.parameters.mode == "ase":
            return _calculate(self,)
        else:
            builder = mapping_calculators[self.ext_out](self)
            from aiida.engine import run_get_node, submit
            #running = run_get_node(builder)
            running = submit(builder)
            self.wchain = running # running[-1] if run_and_get_node
    return wrapper_aiida_trigger

def aiida_post_calculate_trigger(_post_calculate):
    # This wraps the _post_calculate method. 
    @functools.wraps(_post_calculate)
    def wrapper_aiida_trigger(self):
        if self.parameters.mode == "ase":
            return _post_calculate(self,)
        else:
            pass
    return wrapper_aiida_trigger

# Read results.
def aiida_read_results_trigger(read_results):
    # This wraps the read_results method. 
    @functools.wraps(read_results)
    def wrapper_aiida_trigger(self):
        if self.parameters.mode == "ase":
            return read_results(self,)
        else:
            output = None
            if self.ext_out == ".wout":
                output = read_output_file(self, self.wchain.outputs.wannier90.retrieved)
            elif self.ext_out in [".pwo",".kho"]:
                output = read_output_file(self)
                if hasattr(output.calc, 'kpts'):
                    self.kpts = output.calc.kpts
            else:
                output = read_output_file(self)
            if self.ext_out in [".pwo",".wout",".kso",".kho"]:
                self.calc = output.calc
                self.results = output.calc.results
            
    return wrapper_aiida_trigger

# Link calculations and results.
def aiida_link_trigger(link):
    # This wraps the link method of Workflow class. 
    @functools.wraps(link)
    def wrapper_aiida_trigger(self,src_calc, src_path, dest_calc, dest_path):
        if self.parameters.mode == "ase":
            return link(self, src_calc, src_path, dest_calc, dest_path)
        elif src_calc: # if pseudo linking, src_calc = None
                dest_calc.parent_folder = src_calc.wchain.outputs.remote_folder
    return wrapper_aiida_trigger

# get files to manipulate further.
def aiida_get_content_trigger(get_content):
    # This wraps the get_content method of _merge_wannier.py (see Koopmans package)
    @functools.wraps(get_content)
    def wrapper_aiida_trigger(calc, relpath):
        if calc.parameters.mode == "ase":
            return get_content(calc, relpath)
        elif hasattr(calc,"wchain"): 
            if calc.ext_out == ".wout":
                inner_remote_folder=calc.wchain.outputs.wannier90.retrieved
            else:
                inner_remote_folder=None
            filename = "aiida"+calc.ext_out
            return get_output_content(
                calc, filename, 
                mode="r", 
                inner_remote_folder=inner_remote_folder)
    return wrapper_aiida_trigger

# Write file to singlefiledata.
def aiida_write_content_trigger(write_content):
    # This wraps the write_content method of _merge_wannier.py (see Koopmans package)
    @functools.wraps(write_content)
    def wrapper_aiida_trigger(dst_file, merged_filecontents):
        if calc.parameters.mode == "ase":
            return get_content(dst_file, merged_filecontents)
        elif hasattr(calc,"wchain"): 
            return generate_singlefiledata(dst_file, merged_filecontents)
    return wrapper_aiida_trigger

# Specific for dfpt run_calculator definition.
def aiida_dfpt_run_calculator(run_calculator):
    # This wraps the write_content method of _merge_wannier.py (see Koopmans package)
    @functools.wraps(run_calculator)
    def wrapper_aiida_trigger(self, calc):
        if calc.parameters.mode == "ase":
            return run_calculator(calc)
        elif hasattr(calc,"wchain"): 
            return self.run_calculators([calc])
    return wrapper_aiida_trigger

# generating the alphas file
def aiida_write_alphas_trigger(write_alphas):
    # This wraps the write_content method of _merge_wannier.py (see Koopmans package)
    @functools.wraps(write_alphas)
    def wrapper_aiida_trigger(self):
        if self.parameters.mode == "ase":
            return write_alphas()
        else:
            return generate_alpha_singlefiledata(self,)
    return wrapper_aiida_trigger