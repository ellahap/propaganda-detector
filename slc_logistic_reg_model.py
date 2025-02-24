import pandas as pd
import numpy as np
import sys
import os
from scipy.stats import uniform
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


class SLCLOGREG:
    @staticmethod
    def GridSearch(X_train, y_train):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(max_iter=1000))
        ])

        param_grid = {
            'tfidf__ngram_range': [(1,1), (1,2)],  # Unigrams, bigrams, trigrams
            'tfidf__max_df': [0.7, 0.9],  # Ignore very common words
            'tfidf__min_df': [1, 3, 5],  # Ignore rare words
            'tfidf__max_features': [None, 2000, 5000],  # Vocabulary size
            'tfidf__sublinear_tf': [True, False],  # Logarithmic TF scaling
            'tfidf__stop_words': [None, 'english'],  # Stopword removal
            'tfidf__norm': ['l1', 'l2', None],
            'tfidf__lowercase': [True, False],
            'tfidf__analyzer': ['word', 'char_wb'],
            'tfidf__strip_accents': ['ascii', 'unicode', 'none']
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print("Best Parameters:", grid_search.best_params_)
        return grid_search.best_params_
    


    @staticmethod
    def RandomizedSearch(X_train, y_train):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(max_iter=1000))
        ])

        param_distributions = {
            'tfidf__ngram_range': [(1,1), (1,2)],  # Unigrams + Bigrams
            'tfidf__max_df': uniform(0.5, 0.5),  # Values between 0.5 and 1.0
            'tfidf__min_df': [1, 3, 5],  # Ignore very rare words
            'tfidf__max_features': [5000, 10000, None],  # Vocabulary size
            'tfidf__sublinear_tf': [True, False],  # Logarithmic scaling
            'tfidf__stop_words': ['english', None],  # Stopword removal
            'clf__C': uniform(0.1, 1.0),  # Regularization strength (Logistic Regression)
        }

        # Initialize Randomized Search
        random_search = RandomizedSearchCV(
            pipeline, 
            param_distributions, 
            n_iter=100,  # Number of random parameter combinations to try
            cv=5,  # 5-fold cross-validation
            scoring='f1',  # Use F1-score for evaluation
            verbose=1, 
            n_jobs=-1,  # Use all CPU cores
            random_state=42  # For reproducibility
        )

        random_search.fit(X_train, y_train)

        print("Best Parameters:", random_search.best_params_)
        return random_search.best_params_
            

    @staticmethod
    def buildLogReg(df):

        X_train, X_test, y_train, y_test = train_test_split(df['sentence'], df['is_propaganda'], test_size=0.2, random_state=42)


        # GRID SEARCH (TAKES VERY LONG TO COMPUTE)
        # vector_params = SLCLOGREG.GridSearch(X_train, y_train) # find the best possible parameters for tf-idf
        # vectorizer = TfidfVectorizer(**vector_params) # load in parameters to vectorizer

        # RANDOMIZED SEARCH
        # best_params = SLCLOGREG.RandomizedSearch(X_train, y_train)
        # tfidf_params = {k.replace("tfidf__", ""): v for k, v in best_params.items() if k.startswith("tfidf__")}
        # clf_params = {k.replace("clf__", ""): v for k, v in best_params.items() if k.startswith("clf__")}
        # vectorizer = TfidfVectorizer(**tfidf_params) # load in parameters to vectorizer

        # STANDARD SETTINGS
        vectorizer = TfidfVectorizer()

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Train logistic regression classifier
        # clf = LogisticRegression(**clf_params) # For Randomized Search
        clf = LogisticRegression() # For standard or grid search
        clf.fit(X_train_tfidf, y_train)


        y_pred = clf.predict(X_test_tfidf)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

        return vectorizer, clf


if __name__ == "__main__":
    SLCLOGREG.buildLogReg()