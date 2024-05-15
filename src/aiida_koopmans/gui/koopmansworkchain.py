from aiida import orm
from aiida.engine.processes.workchains.workchain import WorkChain
from aiida.engine import calcfunction, workfunction
import numpy as np

from aiida.engine import ToContext

from koopmans.workflows import KoopmansDFPTWorkflow, SinglepointWorkflow

class KoopmansWorkChain(WorkChain):
    """WorkChain to run the koopmans package. Very simple, it is only needed for the GUI.

    Args:
        inputs (orm.Dict): inputs as obtained from loading the json file. 
    """
    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.input('input_dictionary', valid_type=orm.Dict, required=False,)
        spec.input('structure', valid_type=orm.StructureData, required=False,
                   help="needed if we run in the GUI and we relax the structure before.")
        
        spec.outline(
            cls.setup,
            cls.run_process,
            cls.results,
        )

        spec.output("alphas", valid_type=orm.List, required= False)
        spec.output("interpolated_dft", valid_type=orm.BandsData, required=False)
        spec.output("interpolated_koopmans", valid_type=orm.BandsData, required=False)
    
    @classmethod
    def from_json(cls,):
        pass
    
    def setup(self):
        wf = SinglepointWorkflow._fromjsondct(self.inputs.input_dictionary.get_dict())
        self.ctx.workflow = KoopmansDFPTWorkflow.fromparent(wf)

        return 
    
    def run_process(self):
        # for now in the DFPT AiiDA wfl we just run_and_get_node, so no need to have the context.
        self.ctx.workflow._run()
        return
        
    
    def results(self):
        
        parent = orm.load_node(self.ctx.workflow.dft_wchains_pk[0])
        bands_dft = merge_bands(parent.outputs.remote_folder, method="dft")
        bands_koopmans = merge_bands(parent.outputs.remote_folder, method="koopmans")
        
        self.out("interpolated_dft",bands_dft)
        self.out("interpolated_koopmans",bands_koopmans)
        
        return
    

@calcfunction
def merge_bands(remote_pw, method="dft"):
    # I want to have both dft and koopmans method and call this calcfunction once, 
    # but for now it is fine to call it twice.
    # remote_pw is needed to access self (KoopmansWorkChain) in the calcfunction
    workchain = remote_pw.creator.caller.caller
    
    bands = {"dft":[],"koopmans":[]}
    method_loop = "dft"
    for job in workchain.called:
        if job.process_type == "aiida.workflows:wannier90_workflows.bands":
            bands[method_loop].append(job.outputs.band_structure)
        if job.process_type == "aiida.calculations:koopmans":
            method_loop = "koopmans"
            if "eigenvalues" in job.outputs.output_parameters.get_dict().keys():
                bands[method_loop] = job.outputs.output_parameters.get_dict()["eigenvalues"]
                
    for method_merge in [method.value]: 
        if method_merge == "koopmans":
            new_bands_array = bands[method_merge]
        else:
            new_bands_array = bands[method_merge][0].get_bands()
            for i in range(1,len(bands[method_merge])):
                new_bands_array = np.concatenate((new_bands_array,bands[method_merge][i].get_bands()),axis=1)
            
        # Create a band structure object
        merged_bands = bands["dft"][0].clone()
        merged_bands.set_bands(new_bands_array)
        # merged_bands.store()
        #bands[method].append(merged_bands)
        
    return merged_bands