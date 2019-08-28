## General information
All tests have been run on a Ubuntu 18.04 LTS with Tesla P100 16GB. Language is Python 3.

## Environment setup
For environment isolation [miniconda-3](https://docs.conda.io/projects/conda/en/latest/) was the tool of choice. After downloading and installing conda it is advisable to use developers' environment, which is shared [here](https://anaconda.org/yoandinkov/pre-master-thesis-linux).There is a second environment that can be used for Windows users [here](https://anaconda.org/yoandinkov/pre-master-thesis).

## Environment variables
This project uses [dotenv](https://github.com/theskumar/python-dotenv), so you need to have a local .env file in root directory with following properties to run run the code locally:

```
DB_FILE={path to db as a file}
```

* 