## General information
All tests have been run on a Ubuntu 18.04 LTS with Tesla P100 16GB. Language is Python 3.

## Environment setup
For environment isolation [miniconda-3](https://docs.conda.io/projects/conda/en/latest/) was the tool of choice. After downloading and installing conda it is advisable to use researches' environment, which is shared [here](https://anaconda.org/yoandinkov/pre-master-thesis-linux).There is a second environment that can be used for Windows users [here](https://anaconda.org/yoandinkov/pre-master-thesis).

To copy conda environment run following script (example for Linux):
```
conda env create yoandinkov/pre-master-thesis-linux
```

## Build package
Because of the folder structure, an isolated package should be installed. So after importing conda environment locally, go to root folder of repository and run `pip install -e .` this will install _src_ folder as a package and will enable multi level imports.

## Environment variables
This project uses [dotenv](https://github.com/theskumar/python-dotenv), so you need to have a local .env file in root directory with following properties to run the code locally:

```
MONGO_DB='{MongoDB connection string}'
DB_NAME='{MongoDB collection name where articles are stored}'
```

* 