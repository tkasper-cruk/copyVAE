# copyVAE

## Installation

For installation, simply download this repo and run the following commands. **Python >= 3.8** is recommended for running copyVAE.

    cd copyVAE/
    pip install -e .

## Usage

The following command will run the copy number profiling pipeline.

    copyvae path/to/UMI_file path/to/cell_cycle_gene_file

`UMI_file` should be in `.h5ad` format.

Use `Macosko_cell_cycle_genes.txt` file in `./data` folder for cell cycle gene file.

Add flag `-g` for GPU, ex. `-g 1`

### Hyperparameters

`-bin`  Bin size. Default: 25

`-mc` Maximum copy number. Default: 15

`-intd` Hidden layer dimension. Default: 128

`-l` Latent dimension. Default: 15

`-bs` Batch size. Default: 128

`-ep` Number of epochs. Default: 200

`-nc` Number of clones. Default: 2

 #### Example usage with test data:

`copyvae data/BM2_v0s0_n_200.h5ad data/Macosko_cell_cycle_genes.txt -mc 6`

## Outputs
CopyVAE produces the following files in `./output`:

`copy.npy` single-cell pseudo copy numbers in bin resolution.

`clone_#_breakpoints.npy` breakpoints detected for clone #. 

`clone_#_single_cell_profile.npy` single-cell copy number profiles for clone # (in bin resolution).

`clone_#_profile.npy`  clone #'s copy number profile.

See [Wiki](https://github.com/mandichen/copyVAE/wiki) for more details.
