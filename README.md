# Evaluation of KB-BERT for Swedish NLP
<img align="right" width="200" height="330" src="images/bert.png">
This repository is populated with work aiming to investigate the Swedish BERT model (KB-BERT), specifically its performance at named entity recognition and multi-label text classification for news articles and possible improvements to enhance it in the domain.

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

## Data
All input data and various intermediate results can be found [here](https://drive.google.com/drive/u/0/folders/1To6v4SPUL1eHOf2OGtoILg0htD3Mae0E) for reproducability. Placing the data folders locally in the cloned repository will enable all code to run from the get-go. The input data of articles from Bonnier News are retrieved from <i>
data-warehouse-bn.content.article_deduplicated</i> in BigQuery, while the articles from TT are retrieved via the <i>Search API</i> by using get_tt_articles.py.

## Named Entity Recognition (NER)
The <i>ner</i> directory contains work where the pre-finetuned <b>bert-base-swedish-cased-ner</b> is evaluated at NER, as well as exploratory work regarding the applications of NER.

* entity_processing/
    * recognition.py – for performing NER on a dataset of articles, essentially the same code used in the krangy repository
    * cleaning.py – for "cleaning" up the NER output and some initial basic analysis
    * analysis.py – for analyzing the found entities in relation to e.g. article categories
    * category_grouping.py – for grouping entities to categories and vice versa from TT articles
    * clouding.py – for creating a word cloud of the most commonly found entities
* evaluation/
    * nerd_bert_comparison.py – for comparing the performance of nerd and bert using article tags mentioned in the text as labels
    * evaluator.py – for NER evaluation using NER tagged corpuses
    * bert_evaluation.py – for evaluating KB-BERT using the class in evaluator.py
    * threshold_optimization.py – for analyzing how different thresholds for entity confidence affects output and metrics
* similarity/
    * article_similarity – attempt to calculate meaningful similarity between articles base on entities found in the text body
    * category_similarity.py – attempt to calculate meaningful similarity between categories based on entities found in article texts
    * categorize_article.py – attempt to categorize articles based on entities found in the text body
* tt_specific/
    * get_tt_articles.py – for querying and saving TT articles
    * keyword_extraction.py – for extracting keywords (entities) representative of a given IPTC category

If NER is to be performed on new article datasets, the scripts under entity_processing above are generally supposed to be run in the order in which they are listed. The reason being that the output from one script often is used as input to another.

## Multi-Label Text Classification (MLTC)
The <i>mltc</i> directory contains work where <b>bert-base-swedish-cased</b> is finetuned for hierarchical multi-label text classification. Aiming to create a classifier that is able to predict the category/categories of news articles, the models are trained and evaluated on texts from MittMedia and TT, which both employ IPTC subject codes for categorizing. The basis for the finetuned model is heavily inspired by the one described in this [article](https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d).

* create_datasets.py – for creating datasets from TT and MittMedia articles for training and evaluation
* runner.py – for creating, loading and using classifiers
* models.py – contains two types of classification models
* processor.py – for loading article data from a CSV file and transforming it so that it can be inputted to a model
* trainer.py – for training classifiers
* evaluator.py – for evaluating and testing classifiers
* evaluation_analysis.ipynb – presentation/analysis of evaluation results for different model variations

