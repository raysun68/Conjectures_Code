## Main Files

### Adaptive Monte Carlo Search (AMCS) for Sidorenko's Conjecture with graphons:

 `Graphon_demo.py` : Python script implementing AMCS for an initial graphon (you may customize or randomize it) and parameters, works for 4*4 weight matrices. 
 
 `amcs_graphons.py`: Python script containing the AMCS function, including the perturbation strategy.
 
 `helpers.py`: Python script inlcuding many helper functions such as reward functions.

 `PPO_NMCS.py` implements proximal policy optimization for Sidorenko's Conjecture

 `Latin_Search.py` implements optimization for Latin Square graphons in Sidorenko's Conjecture

 Run the code with `python Graphon_demo.py`.

### AMCS for the spectral version of Sidorenko's conjecture:

`Eigen_demo.py`:  Python script implementing AMCS for an initial matrix (you may customize or randomize it) and parameters, works for up to 8*8 matrices.

`Eigen_AMCS.py`: Python script containing the AMCS function, including the perturbation strategy.

 Run the code with `python Eigen_demo.py`.

### AMCS for EFX conjecture: 

`EFX_demo.py`: Python script implementing AMCS for initial utilities, works for 4 agents and 7 items.

`efxcpp.cpp`: C++ script to speed up brute force allocation searching.

Run the code with `python EFX_demo.py`.

### 

###

## Other Files

`Structure_AMCS.qmd` contains matrices that were results of previous optimizations I tried, could be a bit messy.

`Archive` folder contains some old files I shared or byproducts.
