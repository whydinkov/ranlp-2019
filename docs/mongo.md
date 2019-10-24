# Setting up MongoDB
As this research used MongoDB as a primarily db source, it is adviced to migrate the provided version in kaggle into your own MongoDB instance and start from there. This document is giving some starting points on that.

1) Download full dataset from [Kaggle](https://www.kaggle.com/yoandinkov/mediascan-bg-articles)

2) It contains 2 files `articles.json` and `db_dump.zip`

3) Extract `db_dump.zip`

4) Extracted folder `toxic_articles` represents a MongoDB latest data dump performed via [mongodump](https://docs.mongodb.com/manual/reference/program/mongodump/)

5) To use this data dump, you should either install MongoDB locally, on a server, or use their [cloud services](https://cloud.mongodb.com/). Installing and setting up MongoDB is outside the scope of this repository, but [this](https://docs.mongodb.com/manual/installation/) can be used as some initial guidance.

6) Once you've setup your own MongoDB instance (in step 5) you can restore data dump via [mongorestore](https://docs.mongodb.com/manual/reference/program/mongorestore/) command.