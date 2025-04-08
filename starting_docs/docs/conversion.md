# From ASE calculators to AiiDA WorkChains in the DFPT Koopmans workflow - Molecule (no Wannierization)

**In the following, our ASE DFPTWorkflow will be referred as `workflow`**. Indeed, we suppose to have done the following:

```python
from koopmans.io import read
from koopmans.workflows import KoopmansDFPTWorkflow

from koopmans.workflows import DFTPWWorkflow, WannierizeWorkflow

wf = read('aiida-koopmans/example/ozone_dfpt/ozone.json')
workflow = KoopmansDFPTWorkflow.fromparent(wf)
```

The idea is to submit a PwBaseWorkChain when the `calculate` method of the `PwCalculator` (which is implemented in `src/koopmans/calculators/_pw.py`) is triggered by the workflow. The same for kcw.x in the `wann2kc` calculator.

## (1) Telling the workflow that we want to use AiiDA

The first thing is to provide an input in the JSON file in such a way to tell the code to run AiiDA. 
We do this by putting a `mode` input in the file. See the `aiida-koopmans/example/ozone_dfpt/ozone.json` example:

```json
{
  "workflow": {
    "functional": "ki",
    "method": "dfpt",
    "init_orbitals": "kohn-sham",
    "from_scratch": true,
    "alpha_numsteps": 1,
    "pseudo_library": "sg15",
    "mode": {
      "pw_code": "pw-7.2-ok@localhost",
      "kcw_code": "kcw-7.2-ok@localhost",
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
      },
      "metadata_kcw": {    #for now only specific instruction for kcw, but we should add also the others.
          "options": {
            "max_wallclock_seconds": 3600,
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 8,
                "num_cores_per_mpiproc": 1
            },
            "custom_scheduler_commands": "export OMP_NUM_THREADS=1"
        }
      }
    }
  },
  "atoms": {
    "cell_parameters": {
      "vectors": [[5, 0.0, 0.0],
                  [0.0, 5, 0.0],
                  [0.0, 0.0, 5]],
      "units": "angstrom",
      "periodic": false
    },
    "atomic_positions": {
      "units": "angstrom",
      "positions": [
        ["O", 7.0869, 6.0, 5.89],
        ["O", 8.1738, 6.0, 6.55],
        ["O", 6.0, 6.0, 6.55]
      ]
    }
  },
  "calculator_parameters": {
    "ecutwfc": 65.0,
    "ecutrho": 260.0,
    "nbnd": 20
  }
}
```

fundamental are (for now) three keys: `pw_code`, `kcw_code` and `metadata`. This will provide the additional information needed to run AiiDA. Metadata should be multiple, one for each calculation. But for now it is ok like this.

The `mode` input will be treated as a `Setting` in the `src/koopmans/settings/_workflow.py` (line ~ 122) and can be accessed in the workflow under the `workflow.parameters.mode` attribute (settings are stored under the `parameter` attribute of the DFPT workflow).

The `workflow.parameters.mode` is then used in the `_run()` method of the `workflow` to trigger AiiDA things and deactivate the `manual` managing of the directories and data.

## (2) Triggering the DFTPWWorkflow and PWCalculator to run PwBaseWorkChain

We are now at the point to run our first simulation, which is the *scf* calculation for the ozone molecule. This means that, in the `DFPTWorkflow._run`, we are creating the instance of a `DFTPWWorkflow` (line ~ 156), and then we are submitting it via its `run()` method. If you open the method (`src/koopmans/workflows/_dft.py`, line ~ 63), you can see that it generates the calculator via the  `calc=self.new_calculator('pw')` call. This can be inspected in the `src/koopmans/workflows/_workflow.py`, line ~ 611. In principle a `PWCalculator` instance is created and from the AiiDA side we just attach the `mode` attribute to the `calc` here. 

Then,  the `DFTPWWorkflow` runs the PWCalculator via the `run_calculator` method. This will in principle call the `calculate()` method of the PWCalculator (or *calc* in our story), which you can find at line ~ 120 of `src/koopmans/calculators/_pw.py`. 
**Here** I implemented the logic to generate a PwBaseWorkChain instance via the method `get_builder_from_ase()` and then run it. **We just run and not
submit because I want to run without services, for now**. But this can change.

We then store the completed PwBaseWorkChain instance in the `dft_wchain` dictionary (under the key "scf" or f"{calc.parameters.calculation}"), attribute of the calculator, as well as the same attribute but for the whole DFPTWorkflow.

## (3) Analysing results 

This is needed in order to populate the `results` attribute of the PWCalculator, as in the standard ASE-only DFPT. 
We accomplish this still in the `calculate()` method of the PWCalculator, by calling a new `read_results()` method of the same calculator. You can find the original version in the ase-koopmans package: https://github.com/elinscott/ase_koopmans/blob/master/ase/calculators/espresso/_espresso.py.
This behaves as the usual method if `mode=='ase'`, but in case we are using AiiDA, it will just use the `espresso.io.read` function to read the `aiida.out` file with the trick of the `tempfile.TemporaryDirectory()`. 

In this way, we in principle have provided all the needed IO functionalities of the standard Koopmans PWCalculator. 

## (4) wann2kc calculator

This is done in a different way, without a particular reason. The `new_calculator()` method is called in the `DFPTWorkflow._run()` (line ~187) and then we act again in the Calculator, here being the `Wann2KCCalculator` class. 

Here I really override the standard `calculate` method, by using the same trick as pw: we check if the mode is 'ase' or not, if not we call the `from_wann2kc_to_KcwCalculation` (now just a function, in the same file of `Wann2KCCalculator` for simplicity) and we run the calculation, which is a `KcwCalculation`, implemented in the `aiida-koopmans` plugin. 
We then store the calcjob as attribute of the DFPTWorkflow (`DFPTWorkflow.wann2kc_calculation`).

### (4.1) The scf parent folder

We have to provide the scf parent folder. This can be accessed via the `workflow.dft_wchain["scf"].outputs.remote_folder`, and should be stored as `KcwCalculation.inputs.parent_folder`. 
We do this in an hardcoded way at line ~ 688 of `src/koopmans/workflows/_workflow.py`.

## (5) The screening and ham calculations

These are managed moreless in the same way of the wann2kc calculation. We define the corresponding `read_results` and `calculate` methods. I think the builder generator can be generalized among all the KcwCalculation instances.

### the filename when we `read_results()`

The ase.io.read function has some problem in dealing with "aiida.out" filename, so when we read the results we provide the same filename as it would have been decided in standard ASE calc:

```python
with tempfile.TemporaryDirectory() as dirpath:
  # Open the output file from the AiiDA storage and copy content to the temporary file
  for filename in retrieved.base.repository.list_object_names():
      if '.out' in filename:
          # Create the file with the desired name
          readable_filename = "ks.kso"
          temp_file = pathlib.Path(dirpath) / readable_filename
          with retrieved.open(filename, 'rb') as handle:
              temp_file.write_bytes(handle.read())
      
          output = io.read(temp_file)
```


# From ASE calculators to AiiDA WorkChains in the DFPT Koopmans workflow - Solids (Wannierization)

## DFT: scf+nscf
The SCF part is the sameas above, and then we connect with the NSCF part by means of additional logic in `src/koopmans/workflows/_workflow.py`, line ~ 689. in principle we check if the workflow has the `dft_wchains` dictionary and in particular if there is the "scf" one, to them takes its parent folder. 
Now the idea is to create the wannierization WannierBandsWorkChain for each block by using the https://github.com/aiidateam/aiida-wannier90-workflows/blob/main/examples/example_04.py, which starts from a PwBaseWorkChain.

## Separated Wannierization
I submit separated `Wannier90BandsWorkChain`, following the loop on block as in the Koopmans `WannierizeWorkflow`. So the block, projections, are given by input in the JSON file. Each workchain is then store under the `w90_wchains` attribute of the `WannierizeWorkflow` instance. The builder is generated on the fly and stored as attribute of the first calculator which is generated for each block. The calculator is the `Wannier90Calculator` (*src/koopmans/calculators/_wannier90.py*) where I define a `calculate` method in such a way to override the one which is inherited from its parent calculator classes. This calculate method is triggered when we use the `run_calculator` of the `WannierizeWorkflow`. The `read_results` method is defined as in the `PWCalculator` (note the `*wout` extension here).

In the `WannierizeWorkflow`, for each block (i.e. iteration in the loop), I wait for the generation of the first w90 calculator and then I do:
```pyhton
if not self.parameters.mode == "ase":
  builder = get_wannier90bandsworkchain_builder_from_ase(self, calc_w90)
  calc_w90.builder_aiida = builder
  self.run_calculator(calc_w90)
  self.w90_wchains[block.directory.name] = calc_w90.wchain
```
This is defined in *src/koopmans/workflows/_wannierize.py*. We generate and submit an AiiDA `Wannier90BandsWorkChain`, which contains all the logic to obtain final Wannier bands, for each block.

Two issues with the `aiida-wannier90-workflows` package:

-  we should always set `builder.wannier90.shift_energy_windows = False`, otherwise if we have fixed occupation we have an exception (does not find the Fermi energy);
-  not possible to start from nscf, only scf. Actually this is a sort of a bug, because I know that starting from scf allows to set a lot of things automatically, but in our case we want to start from nscf, and not scf, as we need several wannierizations and so it would be a waste of computational time;

**I have done some modification in the aiida-wannier90-workflows to support these features** -> need to make a PR or at least an issue. 



### Merging wannier files and the `produce_wannier90_files` function

The `produce_wannier90_files` function is in the aiida-koopmans/data/utils.py and is used to produce the singlefiledata then input of the KcwCalculation: wannier U matrices, wannier centres.
The merging of files from different wannierization is quite complex but I did some large modifications in the following routines of koopmans:

- `merge_wannier_files` + the one called there.

I store the merged files in the self.wannier90_files dictionary.

