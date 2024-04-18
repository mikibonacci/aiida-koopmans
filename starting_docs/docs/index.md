# The aiida-koopmans plugin for [`AiiDA`](http://www.aiida.net)

Authors: Miki Bonacci (PSI), Julian Geiger (PSI).

This is meant to help the [Koopmans package](https://koopmans-functionals.org/en/latest/index.html) to be used within AiiDA.
For now, nothing is here, as we are mainly working in a [fork](https://github.com/mikibonacci/koopmans) of the Koopmans code.

Every modification done in the Koopmans code by Miki Bonacci is denoted by the comment `#MB mod`. 

The Koopmans code should work as espected when AiiDA is not used.

### Todo:

For now, some method like `get_builder_from_ase` are in the fork of the Koopmans code. However,
the idea is to put them here, once everything works at the 0-th order.

- pseudos from the JSON file (effort: medium-to-high)
- multiple metadata for the different calculations (effort: easy)
- guide on how to start aiida and koopmans@aiida in few minutes (effort: easy-to-medium)
- tests
- link and not hard copy of saves.
- test ASE vs AiiDA for ozone and silicon.
- test also in case you need to merge blocks (ZnO maybe)?
- src/koopmans/calculators/_wann2kc.py why I have to rename the builder.wann inputs.

### Questions:

- Does it matter to parse outputs of the `wann2kc` calculation? it seems that in the workflow is not implemented.
- Every name of the new methods can change, for now we just want a working example which can work smootly, even if the implementation is not the best one.
- We create the AiiDA WorkChain instance when the calculator.run method is called; is it ok? or we should create the instance when the calculator is initialized? and then the workchain is stored as an attribute in the calculator? it seems better to me this second way.
- for now only run, not submit, in the workflow. This because the idea is to run without services, in a first implementation.