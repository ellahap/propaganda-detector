# Propaganda Detector

## About
This project was created for UIUC's [LING 413: Corpus Linguistics](https://courses.illinois.edu/schedule/2025/spring/LING/413).  It is a propaganda detector built upon data collected for the [2019 Workshop on NLP4IF](http://www.netcopia.net/nlp4if/2019/index.html).  You can find the original repo and data [here](https://github.com/marcogdepinto/PropagandaDetection/tree/master?tab=readme-ov-file#pipeline-for-2019-workshop-on-nlp4if-censorship-disinformation-and-propaganda).  You can also find the findings of this shared conference challenge summarized in [this paper](https://aclanthology.org/D19-5024/), and [this paper](https://aclanthology.org/D19-5000/).

An ACL-style research paper detailing my findings is included in the repo above, in the file ["research-paper.pdf"](https://github.com/ellahap/propaganda-detector/blob/main/research_paper.pdf).

## Abstract
This project attempts to create a propaganda detection classifier, trained on a sentence-level human-annotated dataset of news articles. It then explores the relationships between news outlets and predicted propaganda, and media bias and predicted propaganda. The two propaganda detection classifiers built are logistic regression and SVM models with TF-IDF. The logistic regression model performs slightly better, with an accuracy score of 0.75. After applying the logistic regression model to a new dataset of news articles, each news publication’s proportion of articles predicted to contain propaganda is recorded, and there is no correlation observed between media bias and predicted propaganda. The logistic regression model is also tested on a separate dataset of annotated tweets, with an accuracy of 0.72.
