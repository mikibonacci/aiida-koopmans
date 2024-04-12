run the example via "koopmans ozone.json".
delete the 'mode' input in the file to run the standard ASE Koopmans.

To run this via script/notebook:

```python
from koopmans.io import read
from koopmans.workflows import KoopmansDFPTWorkflow

from koopmans.workflows import DFTPWWorkflow, WannierizeWorkflow

wf = read('aiida-koopmans/example/ozone_dfpt/ozone.json')
workflow = KoopmansDFPTWorkflow.fromparent(wf)

new_wfl = workflow._run()
```
