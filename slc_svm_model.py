import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


class SLCSVM:
    @staticmethod
    def GridSearch(X_train_tfidf, y_train):
        # BEST PARAMETERS: 10, linear, scale (Just to save some time!)
        param_grid = {
            'C': [0.1, 1, 10],  # Regularization strength
            'kernel': ['linear', 'rbf'],  # Test both linear and RBF kernel
            'gamma': ['scale', 'auto', 0.01, 0.1],  # Only relevant for RBF kernel
            'degree': [2, 3, 4, 5]
        }

        grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)
        grid_search.fit(X_train_tfidf, y_train)

        print("Best Parameters:", grid_search.best_params_)
        return grid_search.best_params_
    
    @staticmethod
    def RandomizedSearch(X_train_tfidf, y_train):
        param_dist = {
            'C': np.logspace(-2, 2, 10),  # Values between 0.01 to 100
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],  # Only used for 'rbf' and 'poly'
            'degree': [2, 3, 4, 5],  # Only used for 'poly'
            'coef0': np.linspace(0, 1, 5),  # Only used for 'poly' and 'sigmoid'
            'class_weight': [None, 'balanced']
        }

        random_search = RandomizedSearchCV(
            SVC(), param_distributions=param_dist, 
            n_iter=20, cv=5, scoring='f1', verbose=1, n_jobs=-1, random_state=42
        )

        random_search.fit(X_train_tfidf, y_train)
        print("Best Parameters:", random_search.best_params_)
        return random_search.best_params_

    @staticmethod
    def buildSVM(df):

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            df['sentence'], df['is_propaganda'], test_size=0.2, random_state=42
        )

        # Convert text to TF-IDF features
        vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=10000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Grid Search
        # params = SLCSVM.GridSearch(X_train_tfidf, y_train)
        # params = SLCSVM.RandomizedSearch(X_train_tfidf, y_train)


        # Train SVM with best parameters
        pipeline = Pipeline([
            ('tfidf', vectorizer),  # Use the trained TF-IDF vectorizer
            # ('clf', SVC(**params))  # For Grid Search
            ('clf', SVC()) # For Standard settings
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    SLCSVM.buildSVM()
