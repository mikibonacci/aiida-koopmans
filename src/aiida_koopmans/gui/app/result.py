"""Koopmans results view widgets"""
from aiidalab_qe.common.panel import ResultPanel

from aiidalab_qe.common.bandpdoswidget import cmap, get_bands_labeling,BandPdosPlotly

import numpy as np
import json

def replace_symbols_with_uppercase(data):
    symbols_mapping = {
        "$\Gamma$": "\u0393",
        "$\\Gamma$": "\u0393",
        "$\\Delta$": "\u0394",
        "$\\Lambda$": "\u039B",
        "$\\Sigma$": "\u03A3",
        "$\\Epsilon$": "\u0395",
    }

    for sublist in data:
        for i, element in enumerate(sublist):
            if element in symbols_mapping:
                sublist[i] = symbols_mapping[element]
                
def get_bands_from_koopmans(koopmans_output):
    full_data = {
            "dft": None,
            "koopmans": None,
        }
    parameters = {}
    
    dft_bands = koopmans_output.interpolated_dft
    data = json.loads(
        dft_bands._exportcontent("json", comments=False)[0]
    )
    # The fermi energy from band calculation is not robust.
    data["fermi_energy"] = 0
    data["pathlabels"] = get_bands_labeling(data)
    replace_symbols_with_uppercase(data["pathlabels"])

    bands = dft_bands._get_bandplot_data(cartesian=True, prettify_format=None, join_symbol=None, get_segments=True)
    parameters["energy_range"] = {
        "ymin": np.min(bands["y"]) - 0.1,
        "ymax": np.max(bands["y"]) + 0.1,
    }
    data["band_type_idx"] = bands["band_type_idx"]
    data["x"] = bands["x"]
    data["y"] = bands["y"]
    full_data["dft"] = [data, parameters]
    
    koop_bands = koopmans_output.interpolated_koopmans
    data = json.loads(
        koop_bands._exportcontent("json", comments=False)[0]
    )
    # The fermi energy from band calculation is not robust.
    data["fermi_energy"] = 0
    data["pathlabels"] = get_bands_labeling(data)
    replace_symbols_with_uppercase(data["pathlabels"])

    bands = koop_bands._get_bandplot_data(cartesian=True, prettify_format=None, join_symbol=None, get_segments=True)
    parameters["energy_range"] = {
        "ymin": np.min(bands["y"]) - 0.1,
        "ymax": np.max(bands["y"]) + 0.1,
    }
    data["band_type_idx"] = bands["band_type_idx"]
    data["x"] = bands["x"]
    data["y"] = bands["y"]
    full_data["koopmans"] = [data, parameters]
    
    return full_data

class Result(ResultPanel):
    """Result panel for the bands calculation."""

    title = "Koopmans bands"
    workchain_labels = ["koopmans"]

    def __init__(self, node=None, **kwargs):
        super().__init__(node=node, **kwargs)

    def _update_view(self):
        # Check if the workchain has the outputs
        try:
            koopmans_output = self.node.outputs.koopmans
        except AttributeError:
            koopmans_output = None

        bands = get_bands_from_koopmans(self.node.outputs.koopmans)

        fig = BandPdosPlotly(bands_data=bands["koopmans"][0]).bandspdosfigure
        fig_drop = BandPdosPlotly(bands_data=bands["dft"][0]).bandspdosfigure
        fig.add_scatter(y=fig_drop.data[0].y,x=fig_drop.data[0].x, name='DFT')
        del fig_drop

        trace_koopmans = fig.data[0]
        trace_koopmans.name = 'Koopmans'
        trace_koopmans.showlegend = True

        fig.layout.title.text = 'Interpolated Koopmans band structure'
        fig.layout.autosize = True

        self.children = [
            fig,
        ]
