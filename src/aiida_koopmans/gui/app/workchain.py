from aiida_koopmans.gui.koopmansworkchain import KoopmansWorkChain
from aiida import orm

def check_codes(pw_code, pw2wannier90_code, wannier90_code, kcw_code):
    """Check that the codes are installed on the same computer."""
    if (
        not any(
            [
                pw_code is None,
                pw2wannier90_code is None,
                wannier90_code is None,
                kcw_code is None,
            ]
        )
        and len(
            set(
                (
                    pw_code.computer.pk,
                    pw2wannier90_code.computer.pk,
                    wannier90_code.computer.pk,
                    kcw_code.computer.pk,
                )
            )
        )
        != 1
    ):
        raise ValueError(
            "All selected codes must be installed on the same computer. This is because the "
            "Koopmans calculations rely on large files that are not retrieved by AiiDA."
        )

def set_component_resources(code_info):
    """Set the resources for a given component based on the code info."""
    if code_info:  # Ensure code_info is not None or empty
        metadata = {
                "options": {
                    "max_wallclock_seconds": 3600*12,
                    "resources": {
                        "num_machines": code_info["nodes"],
                        "num_mpiprocs_per_machine": code_info["ntasks_per_node"],
                        "num_cores_per_mpiproc": code_info["cpus_per_task"],
                    },
                    "custom_scheduler_commands": "export OMP_NUM_THREADS=1"
                }
            }
        return metadata
                
def get_builder(codes, structure, parameters, **kwargs):
    """Get a builder for the PwBandsWorkChain."""
    
    pw_code = codes.get("pw")["code"]
    pw2wannier90_code = codes.get("pw2wannier90")["code"]
    wannier90_code = codes.get("wannier90")["code"]
    kcw_code = codes.get("kcw")["code"]
    check_codes(pw_code, pw2wannier90_code, wannier90_code, kcw_code)
    
    input_dictionary = parameters["koopmans"].pop("input_dictionary",{})
    
    input_dictionary["workflow"]["mode"]= {
            "pw_code": pw_code.full_label,
            "kcw_code": kcw_code.full_label,
            "pw2wannier90_code": pw2wannier90_code.full_label,
            #"projwfc_code": "projwfc-qe-ki_proj@localhost_koopmans" ,
            "wannier90_code": wannier90_code.full_label,
            "metadata":set_component_resources(codes.get("pw")),
            "metadata_w90":set_component_resources(codes.get("wannier90")),
            "metadata_kcw":set_component_resources(codes.get("kcw")),
          }
    
    kcw_wf = KoopmansWorkChain.get_builder()
    kcw_wf.input_dictionary = orm.Dict(input_dictionary)
    
    return kcw_wf


workchain_and_builder = {
    "workchain": KoopmansWorkChain,
    #"exclude": ("structure", "relax"),
    "get_builder": get_builder,
    #"update_inputs": update_inputs,
}