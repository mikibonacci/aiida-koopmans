# -*- coding: utf-8 -*-
"""Panel for Koopmans bands."""

import ipywidgets as ipw
import json
import pathlib
import tempfile

from aiidalab_qe.common.panel import Panel

class Setting(Panel):
    title = "Koopmans bands"
    identifier = "koopmans"

    def __init__(self, **kwargs):
        self.settings_help = ipw.HTML(
            """<div style="line-height: 140%; padding-top: 0px; padding-bottom: 5px">
            The Koopmans band structure is computed using the AiiDA-enabled version of the <b>
            <a href="https://koopmans-functionals.org/en/latest/"
        target="_blank">Koopmans package</b></a> (E. Linscott et al., 
        <a href="https://pubs.acs.org/doi/10.1021/acs.jctc.3c00652"
        target="_blank">J. Chem. Theory Comput. 2023 <b>19</b>, 20, 2023</a>) and the <a href="https://github.com/mikibonacci/aiida-koopmans"
        target="_blank">aiida-koopmans</b></a> plugin, co-developed by Miki Bonacci, Julian Geiger and Edward Linscott (Paul Scherrer Institut, Switzerland). 
        <br>
        <br>
        For now, we allow one way to provide Koopmans settings, i.e. through the upload button below. You should pass 
        the same file that is needed to run a standard Koopmans@AiiDA simulation, i.e. the codes should be set there, 
        and not in step 3 of the app (this is just a temporary limitation). 
        <br>
        <br>
        Only DFPT workflow is available.
            
            </div>""",
            layout=ipw.Layout(width="400"),
        )
        
        # Upload buttons
        self.upload_widget = ipw.FileUpload(
                    description="Upload Koopmans json file",
                    multiple=False,
                    layout={"width": "initial"},
                )
        self.upload_widget.observe(self._on_upload_json, "value")
        
        self.reset_uploads = ipw.Button(
            description="Discard uploaded file",
            icon="pencil",
            button_style="warning",
            disabled=False,
            layout=ipw.Layout(width="auto"),
        )
        self.reset_uploads.observe(self._on_reset_uploads_button_clicked, "value")
        
        self.children = [
            self.settings_help,
            ipw.HBox(children=[self.upload_widget,self.reset_uploads]),
        ]
        super().__init__(**kwargs)
        
        
    def _on_reset_uploads_button_clicked(self, change):
        self.upload_widget.value.clear()
        self.upload_widget._counter = 0

    def _on_upload_json(self, change):
        # TO BE IMPLEMENTED
        if change["new"] != change["old"]:
            uploaded_filename = next(iter(self.upload_widget.value))
        content = self.upload_widget.value[uploaded_filename]['content']
        self.input_dictionary = json.loads(content.decode('utf-8'))  # Decode content and parse JSON
        print("Uploaded JSON content:")
        print(self.input_dictionary)
                

        
    def get_panel_value(self):
        """Return a dictionary with the input parameters for the plugin."""
        return {
            "input_dictionary": self.input_dictionary,
        }

    def set_panel_value(self, input_dict):
        """Load a dictionary with the input parameters for the plugin."""
        self.input_dictionary = input_dict.get("input_dictionary", {})

    def reset(self):
        """Reset the panel to its default values."""
        self.input_dictionary = {}