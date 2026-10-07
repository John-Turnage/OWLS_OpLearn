# Navier–Stokes dataset

The Navier–Stokes notebooks use the forcing → vorticity dataset of

> J. Kossaifi, N. Kovachki, K. Azizzadenesheli, A. Anandkumar.
> *Multi-Grid Tensorized Fourier Neural Operator for High-Resolution PDEs*, 2023.

It is too large for this repository. Download it from Zenodo,
<https://zenodo.org/records/12825163>, and put these two files in this folder:

```
experiments/navier_stokes/raw_data/nsforcing_train_128.pt   # 10000 forcing/vorticity pairs, 128 x 128 grid
experiments/navier_stokes/raw_data/nsforcing_test_128.pt    #  2000 pairs
```

Each file holds a dictionary with tensors `x` (forcings) and `y` (vorticities at the final
time). They are read with `torch.load(..., weights_only=True)` in `ns_utils.load_ns_data`;
PyTorch is used only for reading these files. The `.pt` files are ignored by git.

You only need the dataset to rerun the experiments (`RECOMPUTE = True`). The figures can be
redrawn from the saved results in `../data/` without it.
