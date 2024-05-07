""" Helper functions for automatically setting up computer & code.
Helper functions for setting up

 1. An AiiDA localhost computer
 2. A "diff" code on localhost

Note: Point 2 is made possible by the fact that the ``diff`` executable is
available in the PATH on almost any UNIX system.
"""

import shutil
import tempfile

from aiida.common.exceptions import NotExistent
from aiida.orm import Code, Computer
from ase.io.espresso import kch_keys, kcp_keys, kcs_keys, pw_keys, w2kcw_keys

LOCALHOST_NAME = "localhost-test"

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

def get_builder_from_ase(pw_calculator):
    from aiida import load_profile, orm
    from aiida_quantumespresso.common.types import ElectronicType
    from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain, PwCalculation

    load_profile()

    """
    We should check automatically on the accepted keywords in PwCalculation and where are. Should be possible.
    we suppose that the calculator has an attribute called mode e.g.

    pw_calculator.mode = {
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
    aiida_inputs = pw_calculator.mode
    calc_params = pw_calculator._parameters
    structure = orm.StructureData(ase=pw_calculator.atoms)

    pw_overrides = {
        "CONTROL": {},
        "SYSTEM": {"nosym": True, "noinv": True},
        "ELECTRONS": {},
    }

    for k in ["calculation", "verbosity"]:  # ,"prefix"
        if k in calc_params.keys():
            pw_overrides["CONTROL"][k] = calc_params[k]

    for k in ["tot_charge", "tot_magnetization", "nbnd", "ecutwfc", "ecutrho", "nspin"]:
        if k in calc_params.keys():
            pw_overrides["SYSTEM"][k] = calc_params[k]

    for k in ["conv_thr"]:
        if k in calc_params.keys():
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

    from aiida_koopmans.calculations.kcw import KcwCalculation

    load_profile()

    builder = KcwCalculation.get_builder()

    wann2kc_control_namelist = w2kcw_keys['control']
    wann2kc_wannier_namelist = w2kcw_keys['wannier']

    control_dict = {
        k: v if k in wann2kc_control_namelist else None
        for k, v in wann2kc_calculator.parameters.items()
    }
    control_dict["calculation"] = "wann2kcw"

    if not any(wann2kc_calculator.atoms.pbc):
        control_dict["assume_isolated"] = "m-t"

    for k in list(control_dict):
        if control_dict[k] is None:
            control_dict.pop(k)

    wannier_dict = {
        k: v if k in wann2kc_wannier_namelist else None
        for k, v in wann2kc_calculator.parameters.items()
    }

    for k in list(wannier_dict):
        if wannier_dict[k] is None:
            wannier_dict.pop(k)

    wann2kcw_params = {
        "CONTROL": control_dict,
        "WANNIER": wannier_dict,
    }

    builder.parameters = orm.Dict(wann2kcw_params)
    builder.code = orm.load_code(wann2kc_calculator.mode["kcw_code"])
    builder.metadata = wann2kc_calculator.mode["metadata"]
    if "metadata_kcw" in wann2kc_calculator.mode:
        builder.metadata = wann2kc_calculator.mode["metadata_kcw"]
    builder.parent_folder = wann2kc_calculator.parent_folder

    if hasattr(wann2kc_calculator, "wannier90_files"):
        builder.wann_u_mat = wann2kc_calculator.wannier90_files["occ"]["u_mat"]
        builder.wann_emp_u_mat = wann2kc_calculator.wannier90_files["emp"]["u_mat"]
        builder.wann_emp_u_dis_mat = wann2kc_calculator.wannier90_files["emp"][
            "u_dis_mat"
        ]
        builder.wann_centres_xyz = wann2kc_calculator.wannier90_files["occ"][
            "centres_xyz"
        ]
        builder.wann_emp_centres_xyz = wann2kc_calculator.wannier90_files["emp"][
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
    }
    control_dict["calculation"] = "ham"

    if not any(kcw_calculator.atoms.pbc):
        control_dict["assume_isolated"] = "m-t"

    for k in list(control_dict):
        if control_dict[k] is None:
            control_dict.pop(k)


    wannier_dict = {
        k: v if k in kch_wannier_namelist else None
        for k, v in kcw_calculator.parameters.items()
    }

    for k in list(wannier_dict):
        if wannier_dict[k] is None:
            wannier_dict.pop(k)

    ham_dict = {
        k: v if k in kch_ham_namelist else None
        for k, v in kcw_calculator.parameters.items()
    }

    for k in list(ham_dict):
        if ham_dict[k] is None:
            ham_dict.pop(k)
        if k == "do_bands":
            ham_dict["do_bands"] = False

    kcw_ham_params = {
        "CONTROL": control_dict,
        "WANNIER": wannier_dict,
        "HAM": ham_dict,
    }

    builder.parameters = orm.Dict(kcw_ham_params)
    builder.code = orm.load_code(kcw_calculator.mode["kcw_code"])
    builder.metadata = kcw_calculator.mode["metadata"]
    if "metadata_kcw" in kcw_calculator.mode:
        builder.metadata = kcw_calculator.mode["metadata_kcw"]
    builder.parent_folder = kcw_calculator.parent_folder

    if hasattr(kcw_calculator, "wannier90_files") and control_dict.get(
        "read_unitary_matrix", False
    ):
        builder.wann_u_mat = kcw_calculator.wannier90_files["occ"]["u_mat"]
        builder.wann_emp_u_mat = kcw_calculator.wannier90_files["emp"]["u_mat"]
        builder.wann_emp_u_dis_mat = kcw_calculator.wannier90_files["emp"]["u_dis_mat"]
        builder.wann_centres_xyz = kcw_calculator.wannier90_files["occ"]["centres_xyz"]
        builder.wann_emp_centres_xyz = kcw_calculator.wannier90_files["emp"][
            "centres_xyz"
        ]

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
    }
    control_dict["calculation"] = "screen"

    for k in list(control_dict):
        if control_dict[k] is None:
            control_dict.pop(k)

    if not any(kcw_calculator.atoms.pbc):
        control_dict["assume_isolated"] = "m-t"

    wannier_dict = {
        k: v if k in kcs_wannier_namelist else None
        for k, v in kcw_calculator.parameters.items()
    }

    for k in list(wannier_dict):
        if wannier_dict[k] is None:
            wannier_dict.pop(k)

    screening_dict = {
        k: v if k in kcs_screening_namelist else None
        for k, v in kcw_calculator.parameters.items()
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
    builder.code = orm.load_code(kcw_calculator.mode["kcw_code"])
    builder.metadata = kcw_calculator.mode["metadata"]
    if "metadata_kcw" in kcw_calculator.mode:
        builder.metadata = kcw_calculator.mode["metadata_kcw"]
    builder.parent_folder = kcw_calculator.parent_folder

    if hasattr(kcw_calculator, "wannier90_files") and control_dict.get(
        "read_unitary_matrix", False
    ):
        builder.wann_u_mat = kcw_calculator.wannier90_files["occ"]["u_mat"]
        builder.wann_emp_u_mat = kcw_calculator.wannier90_files["emp"]["u_mat"]
        builder.wann_emp_u_dis_mat = kcw_calculator.wannier90_files["emp"]["u_dis_mat"]
        builder.wann_centres_xyz = kcw_calculator.wannier90_files["occ"]["centres_xyz"]
        builder.wann_emp_centres_xyz = kcw_calculator.wannier90_files["emp"][
            "centres_xyz"
        ]

    return builder

def get_wannier90bandsworkchain_builder_from_ase(wannierize_workflow, w90_calculator):
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

    nscf = wannierize_workflow.dft_wchains["nscf"]
    aiida_inputs = wannierize_workflow.parameters.mode

    codes = {
        "pw": aiida_inputs["pw_code"],
        "pw2wannier90": aiida_inputs["pw2wannier90_code"],
        "projwfc": aiida_inputs["projwfc_code"],
        "wannier90": aiida_inputs["wannier90_code"],
    }

    builder = Wannier90BandsWorkChain.get_builder_from_protocol(
            codes=codes,
            structure=nscf.inputs.pw.structure,
            pseudo_family="PseudoDojo/0.4/PBE/FR/standard/upf",
            protocol="fast",
            projection_type=WannierProjectionType.ANALYTIC,
        )

    # Use nscf explicit kpoints
    kpoints = orm.KpointsData()
    kpoints.set_cell_from_structure(builder.structure)
    kpoints.set_kpoints(nscf.outputs.output_band.get_array('kpoints'),cartesian=False)
    builder.wannier90.wannier90.kpoints = kpoints

    # set kpath using the WannierizeWFL data.
    k_coords = []
    k_labels = []
    special_k = wannierize_workflow.kpoints.path.todict()["special_points"]
    t=0
    for label in wannierize_workflow.kpoints.path.todict()["labelseq"]:
        k_labels.append([t,label])
        k_coords.append(special_k[label].tolist())
        t=+1
    kpoints_path = orm.KpointsData()
    kpoints_path.set_kpoints(k_coords,labels=k_labels)
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
        position = str(proj["fsite"]).replace("[","").replace("]","").replace(" ","")
        orbital = proj["ang_mtm"]
        converted_proj = "f="+position+":"+orbital
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
    params_pw2wannier90['inputpp']["spin_component"] = "up"
    builder.pw2wannier90.pw2wannier90.parameters = orm.Dict(dict=params_pw2wannier90)


    return builder
