# Evaluation of KB-BERT for Swedish NER
<img align="right" width="200" height="330" src="images/bert.png">
This repository is populated with work aiming to investigate the Swedish BERT model (KB-BERT), specifically its performance at named entity recognition and multi-label text classification for news articles and possible improvements to enhance it in the domain. The exact model evaluated is <b>bert-base-swedish-cased-ner</b>.<br/><br/>

* The original paper on KB-BERT can be found [here](https://arxiv.org/pdf/2007.01658.pdf).
* The KB-BERT models kan be found [here](https://github.com/Kungbib/swedish-bert-models).


## Installation
To create an environment where the code can be run, run the following:
```
git clone https://github.com/BonnierNews/lukas-ner-model
cd lukas-ner-model
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Named Entity Recognition (NER)
The <i>ner</i> directory contains work where the pre-finetuned <b>bert-base-swedish-cased-ner</b> is evaluated at NER, as well as exploratory work on applications of the results.

* /entity_processing – scripts that should be run initially
    * recognition.py – for performing NER on a dataset of articles
    * cleaning.py – for somewhat "cleaning" up the NER output and store in a more convenient format
    * analysis.py – for initial/basic analysis of the NER results
    * clouding.py – for creating a word cloud of the most common entities
* /model_evaluation – evaluation and comparison of KB-BERT and NERD
    * evaluator.py
    * threshold_optimization.py – for analyzing how different thresholds for entity confidence affects output and metrics
* /category_analysis – using entities to calculate similarities
* /tt_specific – work directed at using TT articles
    * get_tt_articles.py – for querying and saving TT articles of specific categories
    * keyword_extraction.py – for extracting keywords (entities) representative of a given category

## Multi-Label Text Classification (MLTC)
The <i>mltc</i> directory contains work where <b>bert-base-swedish-cased</b> is finetuned for hierarchical multi-label text classification. Aiming to create a classifier that is able to predict the category/categories of news articles, the model is trained and evaluated on texts from MittMedia and TT, which both employ IPTC subject codes for categorizing.

* create_datasets.py – Used to create datasets from TT and MittMedia articles for training and evaluation.
* runner.py – Base script for creating, loading and using classifiers.
* models.py – Contains two types of classification models.
* processor.py – Used for loading article data from CSV and transform so that it can be inputted to a model.
* trainer.py – Used for training classifiers
* evaluator.py – Used for evaluating and testing classifiers.

## Data
