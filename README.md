# Detecting toxicity in bulgarian news articles

## Dataset

Contains 237 articles, manually labelled by [Krasimir Gadjokov](https://www.gadjokov.com/) between 2012-2017. As well as 92 non-toxic articles fetched from popular bulgaria media outlets in 2019.

Toxicity categories are as follows:

| Category | Example |
|----------|---------|
| фалшиви новини | [click here](http://bradva.bg/bg/article/article-108980#.WOEh6FPyt3k) |
| клевети | [click here](http://pik.bg/%D0%B1%D0%BE%D0%BC%D0%B1%D0%B0-%D0%B2-%D0%BF%D0%B8%D0%BA-%D1%80%D0%B0%D0%B4%D0%B0%D0%BD-%D0%B8-%D0%BF%D1%80%D0%BE%D1%82%D0%B5%D1%81%D1%82%D0%BD%D0%B0-%D0%BC%D1%80%D0%B5%D0%B6%D0%B0-%D0%BD%D0%B0-%D1%82%D0%B0%D0%B9%D0%BD%D0%B0-%D1%81%D1%80%D0%B5%D1%89%D0%B0-%D0%BF%D0%BB%D0%B0%D0%BD%D0%B8%D1%80%D0%B0%D1%82-%D1%81%D0%B2%D0%B0%D0%BB%D1%8F%D0%BD%D0%B5%D1%82%D0%BE-%D0%BD%D0%B0-%D1%86%D0%B0%D1%86%D0%B0%D1%80%D0%BE%D0%B2-%D0%B4%D0%B0%D0%B2%D0%B0%D1%82-%D0%BF%D0%BE--news363313.html) |
| сензации | [click here](https://fakti.bg/life/234099-3-znaka-che-ste-bogina-v-seksa) |
| реч на омраза | [click here](https://trud.bg/%D1%8F%D0%BA-%D1%80%D0%B8%D1%82%D0%BD%D0%B8%D0%BA-%D0%B7%D0%B0%D0%B1%D0%B8-%D0%BA%D0%B0%D0%B1%D0%B8%D0%BD%D0%B5%D1%82%D1%8A%D1%82-%D0%B2-%D0%B7%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D1%82%D0%B5-%D0%B7%D0%B0/) |
| заблуди | [click here](http://www.zajenata.bg/%D0%BA%D0%B0%D0%BF%D0%B2%D0%B0%D0%B9%D1%82%D0%B5-%D0%BE%D1%82-%D1%82%D0%BE%D0%B7%D0%B8-%D0%BB%D0%B5%D0%BA-%D0%B2-%D1%83%D1%88%D0%B8%D1%82%D0%B5-%D1%81%D0%B8-%D0%B8-%D1%81%D0%BB%D1%83%D1%85%D1%8A%D1%82-%D0%B2%D0%B8-%D1%89%D0%B5-%D1%81%D0%B5-%D0%B2%D1%8A%D0%B7%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%B8-%D0%BD%D0%B0-97!-%D1%82%D0%BE%D0%B7%D0%B8-%D0%BB%D0%B5%D1%81%D0%B5%D0%BD-%D0%BD%D0%B0%D1%82%D1%83%D1%80%D0%B0%D0%BB%D0%B5%D0%BD-%D0%BB%D0%B5%D0%BA-%D0%B5-%D0%B5%D1%84%D0%B8%D0%BA%D0%B0%D1%81%D0%B5%D0%BD-%D0%B4%D0%BE%D1%80%D0%B8-%D0%B7%D0%B0-%D0%B2%D1%8A%D0%B7%D1%80%D0%B0%D1%81%D1%82%D0%BD%D0%B8-%D1%85%D0%BE%D1%80%D0%B0-news81287.html) |
| конспирации | [click here](https://trud.bg/article-4882794/) |
| анти-демократичен | [click here](http://budnaera.com/201701f/17010944.html) |
| про-авториратерен | [click here](http://duma.bg/node/37323) |
| _нетоксичен_ | [click here](https://www.actualno.com/bgfootball/nov-stadion-za-cska-no-ima-seriozni-problemi-za-reshavane-news_737893.html)
Golden labels source: https://mediascan.gadjokov.com/

Distributed in following order:

![distribution](https://user-images.githubusercontent.com/493912/53694505-1f4a1d00-3db0-11e9-9f3b-097a2180eb58.png)


## Models

Approach is to compare different categories of algorithms to conclude if there is any actual improvement with overall classification problem. So models, used are:

 * Baselines
 
 * Classifiers
 
 * SOTA 


## Experiment results


## References
* Gadjokov
* sklearn
* bert
