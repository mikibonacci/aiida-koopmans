# From ASE to PwBaseWorkChain and submission in the DFPT Koopmans workflow

The idea is to submit a PwBaseWorkChain when the `calculate` method of the `PwCalculator` (which is implemented in `src/koopmans/calculators/_pw.py`) is triggered by the workflow.