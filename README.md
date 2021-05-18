# Multifidelity Nonmyopic Active Search

This repository contains code for:

Quan Nguyen, Arghavan Modiri, Roman Garnett. Multifidelity Nonmyopic Active Search. ICML 2021. [INSERT LINK]().

## Requirements

Matlab: [https://www.mathworks.com/products/matlab.html](https://www.mathworks.com/products/matlab.html)

Active learning toolbox: [https://github.com/rmgarnett/active_learning.git](https://github.com/rmgarnett/active_learning.git)

Active search toolbox: [https://github.com/rmgarnett/active_search.git](https://github.com/rmgarnett/active_search.git)

Nonmyopic active search toolbox: [https://github.com/shalijiang/efficient_nonmyopic_active_search](https://github.com/shalijiang/efficient_nonmyopic_active_search)

Drug discovery datasets (not included due to size limit): [https://github.com/rmgarnett/active_virtual_screening.git](https://github.com/rmgarnett/active_virtual_screening.git)
- The generated data should be stored in a folder named `data` at the same directory level as this README file.

Python 3.6+ (for generating tables and figures): [https://www.python.org/](https://www.python.org/)
- Further required libraries are included in `requirements.txt`.

## Running an Experiment

Make sure to create the `data` folder described above and that at least one dataset is present.

To run an experiment, first edit lines 1 - 19 of the file `as_async/run_sim.m` to specify the active search policy, dataset, $k$, $\theta$, and the budget.
Finally, run
```
>> run_sim
```
in Matlab and observe the printed output.

Rerun this file multiple times with different values for the variable `exp` to generate results from repeated experiments.
