# Dataset

Dataset can be downloaded from Kaggle - [https://www.kaggle.com/yoandinkov/mediascan-bg-articles](https://www.kaggle.com/yoandinkov/mediascan-bg-articles). 

It contains 2 files, that are supposed to use in 2 different ways _yet containing the same data_:
- `articles.json` - If you want to kickstart your own research using our data and _do not_ want to extend current repository or to reproduce its experiments, this file is for you.
- `db_dump.zip` - if you plan to reproduce experiments from this repository or to extend existing research on top of this repository, you should use MongoDB instance. The file contains the _snapshot_ version of dataset used during research. It requires additional setup, that is explained in [docs/mongo.md](/docs/mongo.md)
