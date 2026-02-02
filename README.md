# An Optimally Weighted Least Squares Method for Operator Learning  
**Code for Reproducing Numerical Results**

This repository contains all code required to reproduce the numerical experiments in the paper  
**“An Optimally Weighted Least Squares Method for Operator Learning,”** available at https://arxiv.org/pdf/2512.11168.

## Repository Structure
├── Modules/ \
├── Poisson/ \
├── Navier_Stokes/ \
└── Viscous_Burgers/ 

### `Modules/`
Core helper functions and classes used throughout the repository.

Of particular interest:
- **`Block_WLS.py`**  
  Defines the weighted least squares (WLS) object used across all experiments, specifically the block-structured formulation used in our operator-valued regression.
- **`Tensor_Induced_Samp.py`**  
  Implements optimal induced sampling for tensor-product Jacobi polynomial bases.

This directory is intended to be largely problem-agnostic and reusable.

### `Poisson/`, `Navier_Stokes/`, `Viscous_Burgers/`
Each directory contains scripts used to generate the figures and numerical results presented in the paper for the corresponding PDE model. These scripts assemble problem-specific bases, sampling strategies, and data generation pipelines on top of the shared WLS and sampling infrastructure in `Modules/`.

## Reproducibility
Running the scripts in each problem directory reproduces the plots and numerical results reported in the paper (up to randomness in sampling where applicable).

### External Data Sets 
- While the data for the Poisson and Viscous Burgers problems are generated locally, the data for the Navier-Stokes problem was taken from an external data set. These files are too large to be uploaded to git-hub. See **`Navier_Stokes/Spatial_Data/`** for details and download instructions. 
