# Detecting toxicity in news articles - a study for news in Bulgarian language

## Motivation
Current repository is used as an experiment evaluation system. Generated results are being used in a research inspired by [RANLP'19](http://lml.bas.bg/ranlp2019/start.php) conference. 

## Dataset

Contains 221 articles, manually labelled by [Krasimir Gadjokov](https://www.gadjokov.com/) between 2011-2017. As well as 96 non-toxic articles fetched from credible bulgarian news outlets in 2019. To incorporate even more features we use Google API for articles translation. Each article is available in both english and bulgarian.

Toxicity categories are as follows (examples are in Bulgarian):

| Category | Example |
|----------|---------|
| fake news (фалшиви новини) | [click here](http://bradva.bg/bg/article/article-108980#.WOEh6FPyt3k) |
| defamation (клевета) | [click here](http://pik.bg/%D0%B1%D0%BE%D0%BC%D0%B1%D0%B0-%D0%B2-%D0%BF%D0%B8%D0%BA-%D1%80%D0%B0%D0%B4%D0%B0%D0%BD-%D0%B8-%D0%BF%D1%80%D0%BE%D1%82%D0%B5%D1%81%D1%82%D0%BD%D0%B0-%D0%BC%D1%80%D0%B5%D0%B6%D0%B0-%D0%BD%D0%B0-%D1%82%D0%B0%D0%B9%D0%BD%D0%B0-%D1%81%D1%80%D0%B5%D1%89%D0%B0-%D0%BF%D0%BB%D0%B0%D0%BD%D0%B8%D1%80%D0%B0%D1%82-%D1%81%D0%B2%D0%B0%D0%BB%D1%8F%D0%BD%D0%B5%D1%82%D0%BE-%D0%BD%D0%B0-%D1%86%D0%B0%D1%86%D0%B0%D1%80%D0%BE%D0%B2-%D0%B4%D0%B0%D0%B2%D0%B0%D1%82-%D0%BF%D0%BE--news363313.html) |
| sensation (сензация) | [click here](https://fakti.bg/life/234099-3-znaka-che-ste-bogina-v-seksa) |
| hate speech (реч на омраза) | [click here](https://trud.bg/%D1%8F%D0%BA-%D1%80%D0%B8%D1%82%D0%BD%D0%B8%D0%BA-%D0%B7%D0%B0%D0%B1%D0%B8-%D0%BA%D0%B0%D0%B1%D0%B8%D0%BD%D0%B5%D1%82%D1%8A%D1%82-%D0%B2-%D0%B7%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D1%82%D0%B5-%D0%B7%D0%B0/) |
| delusion (заблуда) | [click here](http://www.zajenata.bg/%D0%BA%D0%B0%D0%BF%D0%B2%D0%B0%D0%B9%D1%82%D0%B5-%D0%BE%D1%82-%D1%82%D0%BE%D0%B7%D0%B8-%D0%BB%D0%B5%D0%BA-%D0%B2-%D1%83%D1%88%D0%B8%D1%82%D0%B5-%D1%81%D0%B8-%D0%B8-%D1%81%D0%BB%D1%83%D1%85%D1%8A%D1%82-%D0%B2%D0%B8-%D1%89%D0%B5-%D1%81%D0%B5-%D0%B2%D1%8A%D0%B7%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%B8-%D0%BD%D0%B0-97!-%D1%82%D0%BE%D0%B7%D0%B8-%D0%BB%D0%B5%D1%81%D0%B5%D0%BD-%D0%BD%D0%B0%D1%82%D1%83%D1%80%D0%B0%D0%BB%D0%B5%D0%BD-%D0%BB%D0%B5%D0%BA-%D0%B5-%D0%B5%D1%84%D0%B8%D0%BA%D0%B0%D1%81%D0%B5%D0%BD-%D0%B4%D0%BE%D1%80%D0%B8-%D0%B7%D0%B0-%D0%B2%D1%8A%D0%B7%D1%80%D0%B0%D1%81%D1%82%D0%BD%D0%B8-%D1%85%D0%BE%D1%80%D0%B0-news81287.html) |
| conspiracy (конспирация) | [click here](https://trud.bg/article-4882794/) |
| anti-democratic (анти-демократичен) | [click here](http://budnaera.com/201701f/17010944.html) |
| pro-authoritarian (про-авториратерен) | [click here](http://duma.bg/node/37323) |
| non-toxic (нетоксичен) | [click here](https://www.actualno.com/bgfootball/nov-stadion-za-cska-no-ima-seriozni-problemi-za-reshavane-news_737893.html)

Labels' source of truth: https://mediascan.gadjokov.com/

<img src="https://user-images.githubusercontent.com/493912/62256881-6726ac00-b403-11e9-9060-89f0eebce71f.png" width="400px" />

Dataset can be downloaded from [here](https://www.kaggle.com/yoandinkov/mediascan-bg-articles).

Detailed information about dataset, can be found in [docs](/docs/data.md).

## Features

We have generated following feature sets for both English and Bulgarian:

| Language | Feature set | Title | Text |
|-------|----| ---|-------|
| Bulgarian | BERT | 768 | 768 |
| Bulgarian | LSA  | 15 | 200| 
| Bulgarian | Stylometry | 19 | 6 |
| Bulgarian | XLM | 1024 | 1024 |
| English | BERT | 768 | 768 |
| English | ELMO | 1024 | 1024 |
| English | NELA | 129 | 129 |
| English | USE | 512 | 512 |
|  - |  Media | 6  |


## Experiments

We have conducted experiments by combinding different feature sets, as well as introducing a meta classification. 
Meta classification is based on posterior probablities of other experiments result.
For each experiment setup we use fine-tuned [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). Provided results are avaraged after 5-fold experiment split.

| Language | Feature set | Accuracy | F1-macro| 
|-|-|-|-|
| - | Baseline | 30.30 | 05.17 |
| Bulgarian | BERT(title), BERT(text) | 47.69 | 32.58 |
| Bulgarian | XLM(title), XLM (text)  | 38.50 | 24.58 | 
| Bulgarian | Styl(title), Styl(text) | 31.89 | 08.51 |
| Bulgarian | LSA(title), LSA(text) | 55.59 |42.11 |
| Bulgarian | Bulgarian combined | 39.43 | 24.38 |
| English | USE(title), USE(text) | 53.70 | 40.68 |
| English | NELA(title), NELA(text) | 36.36 |23.04 |
| English | BERT(title), BERT(text) | 52.05 | 39.78 |
| English | ELMO(title), ELMO(text) | 54.60 | 40.95 |
| English | English combined | 42.04 | 15.64 |
| - | Media meta | 42.04 | 15.64 |
| - | All combined | 38.16 | 26.04 |
| - | __Meta classifier__ | __59.06__ | __39.70__| 


## References

## Acknoledgemnets
This research is part of the [Tanbih project](http://tanbih.qcri.org/), which aims to limit the effect of "fake news", propaganda and media bias by making users aware of what they are reading. The project is developed in collaboration between the Qatar Computing Research Institute (QCRI), HBKU and the MIT Computer Science and Artificial Intelligence Laboratory (CSAIL).

This research is also partially supported by Project UNITe BG05M2OP001-1.001-0004 funded by the OP "Science and Education for Smart Growth" and the EU via the ESI Funds.
