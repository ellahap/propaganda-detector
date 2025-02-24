import pandas as pd
from createdataframeslc import CreateDataframeSLC
from slc_logistic_reg_model import SLCLOGREG
from slc_svm_model import SLCSVM
import os
import argparse

class SLC():
    @staticmethod
    def importAndCleanData():
        # Create a dataframe of the articles and their labels
        CreateDataframeSLC.load_sentences_with_labels("./datasets/train-articles/", "./datasets/train-labels-SLC", "output_slc")
        df = pd.read_csv("output_slc.csv")
        print(df.head())

        # CLEAN THE DATA
        # print(df_train.isna().sum()) # 668 na values in sentence
        # print(df_train.dtypes) # checks out, article_id: int64, line: int64, is_propaganda: object, sentence: object
        df = df.dropna()
        df.replace('propaganda', 1, inplace=True)
        df.replace('non-propaganda', 0, inplace=True)
        print(df.shape)

        return df
    
    @staticmethod
    def buildModel(model):
        df = SLC.importAndCleanData()

        if model == "LogReg":
            print("Building Logistic Regression model")
            return SLCLOGREG.buildLogReg(df)
        elif model == "SVM":
            print("Building SVM model")
            return SLCSVM.buildSVM(df)
        else:
            print("Error: Invalid entry.")

    

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("model", help="LogReg or SVM model")
    ARGS = PARSER.parse_args()
    LOAD_ARTICLES = SLC.buildModel(ARGS.model)