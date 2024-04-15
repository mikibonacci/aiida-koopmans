run the example via "koopmans si.json".
delete the 'mode' input in the file to run the standard ASE Koopmans.

To run this via script/notebook:

```python
from koopmans.io import read
from koopmans.workflows import KoopmansDFPTWorkflow

from koopmans.workflows import DFTPWWorkflow, WannierizeWorkflow

wf = read('aiida-koopmans/example/silicon_dfpt/si.json')
workflow = KoopmansDFPTWorkflow.fromparent(wf)

new_wfl = workflow._run()
```

