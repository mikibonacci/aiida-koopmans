# from aiidalab_qe.bands.result import Result
from aiidalab_qe.common.panel import OutlinePanel

from .result import Result
from .setting import Setting
from .workchain import workchain_and_builder

from aiidalab_qe.common.widgets import QEAppComputationalResourcesWidget

class Outline(OutlinePanel):
    title = "Koopmans electronic band structure"
    help = """Koopmans DFPT workflow"""

pw2wannier90_code = QEAppComputationalResourcesWidget(
    description="pw2wannier90.x",
    default_calc_job_plugin="quantumespresso.pw2wannier90",
)

wannier90_code = QEAppComputationalResourcesWidget(
    description="wannier90.x",
    default_calc_job_plugin="wannier90.wannier90",
)

kcw_code = QEAppComputationalResourcesWidget(
    description="kcw.x",
    default_calc_job_plugin="koopmans",
)

property = {
    "outline": Outline,
    "code": {
        "pw2wannier90": pw2wannier90_code, 
        "wannier90": wannier90_code, 
        "kcw": kcw_code,
        },
    "setting": Setting,
    "result": Result,
    "workchain": workchain_and_builder,
}