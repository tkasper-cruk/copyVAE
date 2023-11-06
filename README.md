# copyVAE

### Installation

For installation, simply download this repo and run the following commands. **Python >= 3.8** is recommended for running copyVAE.

    cd copyVAE/
    pip install -e .

### Usage

The following command will run the copy number profiling pipeline.

    copyvae path/to/UMI_file path/to/cell_cycle_gene_file
Add flag `-g` for GPU

Use `Macosko_cell_cycle_genes.txt` file in `data` folder for cell cycle gene file.

See [Wiki](https://github.com/mandichen/copyVAE/wiki) for more details.
