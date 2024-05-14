# from aiidalab_qe.bands.result import Result
from aiidalab_qe.common.panel import OutlinePanel

from .result import Result
from .setting import Setting
from .workchain import workchain_and_builder


class Outline(OutlinePanel):
    title = "Koopmans electronic band structure"
    help = """Koopmans DFPT workflow"""

# for now, no codes are provided.

property = {
    "outline": Outline,
    "setting": Setting,
    "result": Result,
    "workchain": workchain_and_builder,
}