# DFPT workflow

Example on ZnO is [here](https://koopmans-functionals.org/en/latest/tutorials/tutorial_3.html).
Remember to set:
```bash
export PARA_PREFIX="mpirun -np 4" 
export OMP_NUM_THREADS=1
```

```bash
(quantum-espresso-7.2) jovyan@6f222c4867e7:~/codes/koopmans/tutorials/tutorial_3$ koopmans zno.json 
  _
 | | _____   ___  _ __  _ __ ___   __ _ _ __  ___
 | |/ / _ \ / _ \| '_ \| '_ ` _ \ / _` | '_ \/ __|
 |   < (_) | (_) | |_) | | | | | | (_| | | | \__ \ 
 |_|\_\___/ \___/| .__/|_| |_| |_|\__,_|_| |_|___/
                 |_|

 Koopmans spectral functional calculations with Quantum ESPRESSO

 version 1.0.1

 Written by Edward Linscott, Riccardo De Gennaro, and Nicola Colonna

 Please cite the papers listed in zno.bib in work involving this calculation

   Wannierization
   ==============
    Running wannier/scf... done
    Running wannier/nscf... done
    Running wannier/block_1/wann_preproc... done
    Running wannier/block_1/pw2wan... done
    Running wannier/block_1/wann... done
    Running wannier/block_2/wann_preproc... done
    Running wannier/block_2/pw2wan... done
    Running wannier/block_2/wann... done
    Running wannier/block_3/wann_preproc... done
    Running wannier/block_3/pw2wan... done
    Running wannier/block_3/wann... done
    Running wannier/block_4/wann_preproc... done
    Running wannier/block_4/pw2wan... done
    Running wannier/block_4/wann... done
    Running wannier/block_5/wann_preproc... done
    Running wannier/block_5/pw2wan... done
    Running wannier/block_5/wann... done
    Running wannier/bands... done
    Running pdos/projwfc... done

  Conversion to Koopmans format
  -----------------------------
   Running wannier/kc... done

  Calculation of screening parameters
  ===================================
   Running screening/kc... done

  Construction of the Hamiltonian
  ===============================
   Running hamiltonian/kc... done

 Workflow complete
 ```


### (1) SCF/Wannierization/projwfc step

 It includes several tasks:
 

#### (1.1) scf+nscf

Computes the electronic density and wavefunctions. 
The `outdir` is the `TMP` where we run the code, and the `pseudo_dir` is a default one once we decided the pseudo to be used. 
Some pseudos are shipped with the package.


#### (1.2) wannier on different blocks

This is a 5 step calculation, where we divided the occupied manifold in 4 parts, following the pdos results (done before) and 1 empty state manifold.
We use the `exclude_bands` parameters to select our bands.
How to automate this? wise analyisis of pDOS.

Questions:
- can you exclude bands really? or you can only use the parameter to split?
- can you split already in aiida-wannier90-workflows? ask Junfeng



#### (1.3) interpolation of bands

Need to understand how the different wannierizations of the occupied bands are put together here;
the folder `wannier/bands` is then disappeared.
This is only the DFT band interpolation.
*I dont think this is fundamental right now, as we just need the wann matrices*

#### (1.4) projwfc


*I dont think this is fundamental right now, as we just need the wann matrices*

### (2) Conversion to Koopmans format

Done in `wannier/kc` but it disappeared. Basically, this step should be done in the KcWCalculation or in a PwToKcwCalculation/calcfunction, 
copying from the from the output of the previous bands calculation remote folder. It runs `kcw.x` code. 
The directory where the dft/wannier data are is the TMP as before.

Questions
- how does this conversion happens? 
- seedname wann or is it ok also aiida?

### (3) calculation of screening parameters

In screening/kc. Runs `kcw.x` code. 
**Important**: it can be skipped if we provide suggestion for each alpha parameters, as in the `tutorial_3`.

in wannier:
```bash
(quantum-espresso-7.2) jovyan@6f222c4867e7:~/work/koopmans_calcs/tutorial_3/hamiltonian$ ls ../wannier/
bands.pwi  block_2  block_5  kc.w2ko   occ      wann_centres.xyz      wann_emp_u.mat
bands.pwo  block_3  emp      nscf.pwi  scf.pwi  wann_emp_centres.xyz  wann_u.mat
block_1    block_4  kc.w2ki  nscf.pwo  scf.pwo  wann_emp_u_dis.mat
```

More in detail:
```bash
(base) jovyan@6f222c4867e7:~/work/koopmans_calcs/bands_Si/wannier$ ls -ltr 
total 396
-rw-r--r-- 1 jovyan users   823 Mar  7 09:09 scf.pwi
-rw-r--r-- 1 jovyan users 43911 Mar  7 09:09 scf.pwo
-rw-r--r-- 1 jovyan users   907 Mar  7 09:09 nscf.pwi
-rw-r--r-- 1 jovyan users 92456 Mar  7 09:09 nscf.pwo
drwxr-sr-x 2 jovyan users  4096 Mar  7 09:09 block_1
drwxr-sr-x 2 jovyan users  4096 Mar  7 09:10 block_2
lrwxrwxrwx 1 jovyan users     7 Mar  7 09:10 occ -> block_1
lrwxrwxrwx 1 jovyan users     7 Mar  7 09:10 emp -> block_2
-rw-r--r-- 1 jovyan users  2595 Mar  7 09:10 bands.pwi
-rw-r--r-- 1 jovyan users 62159 Mar  7 09:10 bands.pwo
lrwxrwxrwx 1 jovyan users    14 Mar  7 09:10 wann_u.mat -> occ/wann_u.mat
lrwxrwxrwx 1 jovyan users    14 Mar  7 09:10 wann_emp_u.mat -> emp/wann_u.mat
lrwxrwxrwx 1 jovyan users    18 Mar  7 09:10 wann_emp_u_dis.mat -> emp/wann_u_dis.mat
lrwxrwxrwx 1 jovyan users    20 Mar  7 09:10 wann_emp_centres.xyz -> emp/wann_centres.xyz
lrwxrwxrwx 1 jovyan users    20 Mar  7 09:10 wann_centres.xyz -> occ/wann_centres.xyz
-rw-r--r-- 1 jovyan users   588 Mar  7 09:10 kc.w2ki
-rw-r--r-- 1 jovyan users 54302 Mar  7 09:10 kc.w2ko
```

in hamiltonian (kc):
```bash
(quantum-espresso-7.2) jovyan@6f222c4867e7:~/work/koopmans_calcs/tutorial_3/hamiltonian$ ls
file_alpharef_empty.txt  kc.kcw_hr.dat      kc.khi            wann_emp_centres.xyz  wann_u.mat
file_alpharef.txt        kc.kcw_hr_emp.dat  kc.kho            wann_emp_u_dis.mat
kc.kcw_bands.dat         kc.kcw_hr_occ.dat  wann_centres.xyz  wann_emp_u.mat
```

Basically, I copy the following files: wann_centres.xyz, wann_u.mat, wann_emp_centres.xyz,  wann_emp_u.mat and wann_emp_u_dis.mat.

####  How do I read and write the alphas? read from screening calc, and write for ham calc?
####  Well, not needed! just use the kc.alpha.dat file, created in the TMP/kcw/.
```

### (4) KC hamiltonian

Runs `kcw.x` code. The corresponding save is kc_kcw.save. Energy units is Hartree. The corresponding KS results are in `kc.save`.


## How this steps are done in the `koopmans` package: the `DFPTWorkflow._run()` method

Basically, the `DFPTWorkflow` is implemented in `src/koopmans/workflows/_koopmans_dfpt.py`. When
you use the *cli* `src/koopmans/cli/main.py`, you are running:

```python
from koopmans.io import read
# Reading in JSON file
workflow = read(args.json)

# Run workflow
workflow.run()
```

where the method `run` is defined in the subclass `Workflow` of the DFTP one, and essentially calls the `_run` method of the specific `DFPTWorkflow`.
This method contains the logic needed to run the entire workflow. Here is where to look when we are looking for a sort of CWL. Each step maybe a single AiiDA workchain and joined together via WorkTree. 


### Specific crucial parts of the code

#### The parallel command setting

PARA_PREFIX=... where is set up in the code? check for PARA_PREFIX in the code... it checks os.environ['PARA_PREFIX']

#### 1 - IF W90 is not required (0D): DFTPWWorkflow

DFTPWWorkflow is called instead of the WannierizerWorkflow if the system is 0D. 
Only scf is done, no nscf as instead is done in w90.

##### scf

```python
# Run PW
self.print('Initialization of density and variational orbitals', style='heading')

# Create the workflow
pw_workflow = DFTPWWorkflow.fromparent(self)

# Update settings
pw_params = pw_workflow.calculator_parameters['pw']
pw_params.nspin = 2
pw_params.tot_magnetization = 0

# Run the subworkflow
with utils.chdir('init'):
    pw_workflow.run()
```

then, to create the pw calculator: 

```python
class DFTPWWorkflow(DFTWorkflow):

    def _run(self):

      # Create the calculator
      calc = self.new_calculator('pw')

      # Update keywords
      calc.prefix = 'dft'
      calc.parameters.ndr = 50
...
```

==========================
#### 1 - WannierizeWorkflow - DFTPWWorkflow

needs to run both scf and nscf



