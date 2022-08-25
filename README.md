[![.github/workflows/test.yml](https://github.com/salarian/py_app_base/actions/workflows/test.yml/badge.svg)](https://github.com/salarian/py_app_base/actions/workflows/test.yml)

# Developer notes
Don't forget to update the name of the project:
- In README.md, replace {Project Name} with the correct name
- In [environment.yml](environment.yml), replace `py_app_base` with the correct name
- In [test.yml](.github/workflows/test.yml), replance `py_app_base` with the correct name
- Fix the links for the github workflow badges on top of README.md


## Conda environment
The python development environment for the project is based on conda. If you don't already have a conda installation, please follow the instructions [here](https://conda.io/miniconda.html) to download and install **Miniconda**.

### Creating conda environment
For the first use, the conda environment can be created using:

```bash
$ conda env create -f environment.yml
```

### Activating conda environment
Don't forget to activte the environment before running the scripts or the unit tests

```bash
$ conda activate {Project Name}
```

### Updateing conda environment
In case the [environment.yml](environment.yml) file has changed, the conda environment can be updated using:

```bash
$ conda env update --name {Project Name} --file environment.yml --prune
```

## Unit tests
Unit tests are based on `pytest`. To run all tests use:

```bash
$ python -m pytest
```
