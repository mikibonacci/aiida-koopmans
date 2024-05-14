from aiida_koopmans.gui.koopmansworkchain import KoopmansWorkChain
from aiida import orm

def get_builder(codes, structure, parameters, **kwargs):
    """Get a builder for the PwBandsWorkChain."""
    kcw_wf = KoopmansWorkChain.get_builder()
    kcw_wf.input_dictionary = orm.Dict(parameters["koopmans"].pop("input_dictionary",{}))
    return kcw_wf


workchain_and_builder = {
    "workchain": KoopmansWorkChain,
    #"exclude": ("structure", "relax"),
    "get_builder": get_builder,
    #"update_inputs": update_inputs,
}