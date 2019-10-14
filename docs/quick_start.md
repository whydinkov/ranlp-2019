# Quick start guide

If you plan to reproduce experiments or to continue research on top of this one following steps should be sufficient to be up and running.

0) Prerequisites
* Import MongoDB to your own instance. Instructions [here](/mongo.md)

1) copy conda environment via  
`conda env create yoandinkov/pre-master-thesis-linux`

2) clone git repository  
`git clone git@github.com:yoandinkov/ranlp-2019.git`

3) once environment is activated and you're in the root folder, you need to install the local package (helps with python references between folders/modules)  
`pip install -e .`

4) once this is done, there is one missing package  
`pip install imblearn`

5) and then you should rename the attached file to .env and paste it in the src/ folder of the repository (the following command is just for test purpouses in linux)  
`~/ranlp-2019/src$ cat .env` 

6) finally, you should be able to execute the experiments  with following commands:  
`python src/experiments/logistic_regression/feature_comparison.py`
`python src/experiments/logistic_regression/feature_combination.py`