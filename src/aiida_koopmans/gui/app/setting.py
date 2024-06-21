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
        For now, we allow two ways to provide Koopmans settings:
        <br>
        (1) setting the following options
        <br>
        (2) through  the upload button below. 
        <br>
        For option (2), you should pass the same file that is needed to run a standard Koopmans simulation.
        <br>
        <br>
        Only "DFPT" method and "KI" functional are currently available. Norm-conserving pseudopotentials must be used.
            </div>""",
            #layout=ipw.Layout(width="400"),
        )
        
        layout_inside_Vboxes = ipw.Layout(width="90%") # it is the default, no need to set it up actually.
        layout_Vboxes = ipw.Layout(width="12.5%")
        
        # Method button
        self.method = ipw.RadioButtons(
            options=['DFPT','deltaSCF'],
            value='DFPT',
            layout=layout_inside_Vboxes
        )
        method_box = ipw.VBox(
            children = [ipw.HTML('Method'),self.method],
            layout=layout_Vboxes
            
        )
        
        # Functional button
        self.functional = ipw.RadioButtons(
            options=['ki','kipz'],
            value='ki',
            layout=layout_inside_Vboxes
        )
        functional_box = ipw.VBox(
            children = [ipw.HTML('Functional'),self.functional],
            layout=layout_Vboxes
        )
        
        # Init orbitals dropdown
        self.init_orbitals_dropdown = ipw.Dropdown(
                    options=['mlwfs','kohn-sham'],
                    value='mlwfs',
                    layout=layout_inside_Vboxes
                )
        init_orbitals_box = ipw.VBox(
            children = [ipw.HTML('Initial orbitals'),self.init_orbitals_dropdown],
            layout=layout_Vboxes
        )

        # Compute alpha checkbox
        self.compute_alpha = ipw.Checkbox(
            indent=False,
            layout=ipw.Layout(flex='0 1 auto', width="10%"),
        )
        compute_alpha_box = ipw.VBox(
            children = [ipw.HBox([
                ipw.HTML("""<div style="line-height: 140%; padding-top: 0px; padding-bottom: 5px">
                         Compute screening alpha <br>
                        (or provide it in the json file)
                        </div>
                        """,
                        layout=ipw.Layout(flex='0 1 auto', width="28%")),
                self.compute_alpha,
                ],),
                
                ],
            layout=ipw.Layout(display='flex', flex_flow='column', align_items='stretch', width='62.5%')
        )
        
        # Full box of the above Koopmans inputs.
        mandatory_inputs_box = ipw.HBox(
            children=[
                method_box,
                functional_box,
                init_orbitals_box,
                compute_alpha_box,
            ],
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
        upload_widget_box = ipw.HBox(
            children=[
                self.upload_widget,
                self.reset_uploads
                ]
            )
        
        self.children = [
            self.settings_help,
             ipw.VBox(
        children=[
           mandatory_inputs_box,
           ipw.HTML("""<div style="line-height: 140%; padding-top: 0px; padding-bottom: 5px">
                         <b>Upload your Koopmans json file to define/override inputs:</b>
                         it is also possible to upload a specific json file with all (or part) of the needed settings.
                         These will override the current options and, if there, also parameters of the simulations to be submitted.
                        </div>                        
                """),
           upload_widget_box,
                ],
            ) 
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
        """
        print("Uploaded JSON content:")
        print(self.input_dictionary)
        """        

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