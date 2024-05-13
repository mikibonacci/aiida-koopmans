run the example via "koopmans si.json".
delete the 'mode' input in the file to run the standard ASE Koopmans, or use the si_no_aiida.json version

To run this examples, for now you need a particular pseudo family installed: `aiida-pseudo install pseudo-dojo -r FR -f upf`.

```bash
conda activate codes #venv where I have installed the pw.x, pw2wannier90.x, kcw.x, wannier90, rsync
export PARA_PREFIX="mpirun -np 8" 
export OMP_NUM_THREADS=1
PATH=/home/jovyan/codes/wannier90:/home/jovyan/codes/q-e-kcw/bin:$PATH

koopmans si_no_aiida.json
```

To run this via script/notebook:

```python
from koopmans.io import read
from koopmans.workflows import KoopmansDFPTWorkflow

from koopmans.workflows import DFTPWWorkflow, WannierizeWorkflow

wf = read('aiida-koopmans/example/silicon_dfpt/si.json')
workflow = KoopmansDFPTWorkflow.fromparent(wf)

new_wfl = workflow._run()
```

