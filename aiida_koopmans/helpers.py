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
            "pseudo_family": "PseudoDojo/0.4/PBE/FR/standard/upf",
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
