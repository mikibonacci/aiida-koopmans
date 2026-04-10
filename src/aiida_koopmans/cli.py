"""
Command line interface (cli) for aiida_koopmans.

Register new commands either via the "console_scripts" entry point or plug them
directly into the 'verdi' command by using AiiDA-specific entry points like
"aiida.cmdline.data" (both in the setup.json file).
"""

import sys

import click

from aiida.cmdline.commands.cmd_data import verdi_data
from aiida.cmdline.params.types import DataParamType
from aiida.cmdline.utils import decorators
from aiida.orm import QueryBuilder
from aiida.plugins import DataFactory


# See aiida.cmdline.data entry point in setup.json
@verdi_data.group("aiida-koopmans", context_settings={'help_option_names': ['-h', '--help']})
def data_cli():
    """Command line interface for aiida-koopmans"""


@data_cli.command("explore")
@click.argument("step_data_pkl", metavar="IDENTIFIER", type=str, required=False, default="step_data.pkl")
def explore(step_data_pkl):
    """Explore the aiida-koopmans step_data.pkl
    
    This is the pickle file produce at runtime by 
    the koopmans ASE workflow when AiiDA engine is used.

    step_data_pkl is the file containing the steps
    and their parameters, which is produced by the ASE workflow.
    If not specified, it defaults to 'step_data.pkl'.
    """
    import dill as pickle
    with open(step_data_pkl, 'rb') as f:
        data = pickle.load(f)
    
    for k,v in data["steps"].items():
        print(f"{k}: {v}")

@data_cli.command("kcw_ham_res")
@click.argument("step_data_pkl", metavar="IDENTIFIER", type=str, required=False, default="step_data.pkl")
def kcw_ham_res(step_data_pkl):
    """Explore the aiida-koopmans step_data.pkl
    
    This is the pickle file produce at runtime by 
    the koopmans ASE workflow when AiiDA engine is used.

    step_data_pkl is the file containing the steps
    and their parameters, which is produced by the ASE workflow.
    If not specified, it defaults to 'step_data.pkl'.
    """
    import dill as pickle
    with open(step_data_pkl, 'rb') as f:
        data = pickle.load(f)
    
    for k,v in data["steps"].items():
        if "kcw_ham" in k:
            import subprocess
            # run a command to get the results
            command = subprocess.run(
                ["verdi", "calcjob", "outputcat", str(v["workchain"])],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if command.returncode == 0:
                # Filter output for lines containing 'highe'
                output_lines = command.stdout.decode('utf-8').splitlines()
                filtered = [line for line in output_lines if "highe" in line]
                print(f"Results for {k}, pk= {v['workchain']}:")
                print("\n".join(filtered))
            else:
                print(f"Error running command for {k}: {command.stderr.decode('utf-8')}")

@data_cli.command("remove")
@click.argument("string", metavar="IDENTIFIER", type=str)
def remove(string):
    """Remove keys in the aiida-koopmans step_data.pkl
    """
    import dill as pickle
    with open('step_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    for k in list(data["steps"].keys()):
        if string in k:
            data["steps"].pop(k,None)
    
    with open('step_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Removed step_data.pkl processes containing '{string}'")


@data_cli.command("list")
@decorators.with_dbenv()
def list_():  # pylint: disable=redefined-builtin
    """
    Display all DiffParameters nodes
    """
    raise NotImplementedError("This command is not implemented yet")
    # DiffParameters = DataFactory("koopmans")

    # qb = QueryBuilder()
    # qb.append(DiffParameters)
    # results = qb.all()

    # s = ""
    # for result in results:
    #     obj = result[0]
    #     s += f"{str(obj)}, pk: {obj.pk}\n"
    # sys.stdout.write(s)


@data_cli.command("export")
@click.argument("node", metavar="IDENTIFIER", type=DataParamType())
@click.option(
    "--outfile",
    "-o",
    type=click.Path(dir_okay=False),
    help="Write output to file (default: print to stdout).",
)
@decorators.with_dbenv()
def export(node, outfile):
    """Export a DiffParameters node (identified by PK, UUID or label) to plain text."""
    raise NotImplementedError("This command is not implemented yet")
    # string = str(node)

    # if outfile:
    #     with open(outfile, "w", encoding="utf8") as f:
    #         f.write(string)
    # else:
    #     click.echo(string)
