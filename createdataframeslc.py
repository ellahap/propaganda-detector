"""
This file creates the dataframe for the SLC task.
"""

import os
import argparse
import pandas as pd
from createlabelsdataframeslc import CreateLabelsDataframeSLC


class CreateDataframeSLC:
    '''
    Task SLC
    The format of a tab-separated line of the gold label and the submission files for task SLC is:
    article_id   sentence_id    label
    where article_id and sentence_id are the identifiers of the article and the sentence
    (the first sentence has id 1) and label={propaganda/non-propaganda}.
    Gold and submission files must have the same number of rows as the number of sentences,
    i.e. of lines, in the article. In order to help participants preparing a submission, we provide
    template prediction files, which have the same format of the gold files where label
    is replaced with ?. Sentences are splitted using dots.
    '''
    @staticmethod
    def load_sentences_with_labels(path: str, path_to_labels: str, savepath: str):

        '''
        This function will be used to create the Pandas dataframe.
        It will also pickle the dataframe to use it later.
        :param path: str
        :param path_to_labels: str
        :param savepath: str
        :return: Pandas dataframe
        '''

        lst = []
        newlst = []
        idx = 1  # Index will be used to count the lines on each article

        print("Creating the labels dataframe..")
        # Creating the labels dataframe using the helper function
        labels_df = CreateLabelsDataframeSLC.load_labels(path_to_labels)

        # print("Labels DataFrame Preview:")
        # print(labels_df.head())
        # print("Labels DataFrame Shape:", labels_df.shape)


        print("Looping through the articles")
        # Creating the series to be used to match the indexes of the single words within the loop

        for dirs, subdir, files in os.walk(path):  # Go through the directory with the files
            for file in files:
                filepath = dirs + '/' + file
                article_id = str(file[:-4][7:])
                with open(filepath, 'r') as single_file:
                    # loop through all lines using f.readlines() method
                    for line in single_file.readlines():
                        # this is how you would loop through each letter
                        line = line.strip().split('\t')
                        lst.append([line, article_id, idx])
                        idx += 1
                    idx = 1 # Index back to one for new line on new article

        # print(lst)
        str_list = [x for x in lst if x[0] != ['']] # Removing the empty lists from the dataset
        # print(str_list)


        # Merging the labels dataframe with the sentences created in the previous loop
        sentences = pd.DataFrame(str_list, columns=['sentence', 'article_id', 'line'])
        # print("Sentences DataFrame Preview:")
        # print(sentences.head())
        # print("Sentences DataFrame Shape:", sentences.shape)


        for index, tok in labels_df.iterrows():
            # Get word data
            identifier = tok['article_id']
            line = tok['line']
            is_propaganda = tok['is_propaganda']

            newlst.append([identifier, line, is_propaganda])
            # print(newlst)

        dataframe = pd.DataFrame(newlst, columns=['article_id', 'line', 'is_propaganda'])
        dataframe['line'] = pd.to_numeric(dataframe['line'])  # Line to numeric for the merge below
        # print(df)

        final_df = dataframe.merge(sentences, on=['line', 'article_id'], how='left')
        # print(final_df)

        # Create a pickle of the dataframe
        print('Saving dataframe..')

        # Save as pickle
        final_df.to_pickle(savepath + ".pkl")

        # Save as CSV
        csv_path = savepath + ".csv"
        final_df.to_csv(csv_path, index=False, encoding="utf-8")

        print(f"Completed. Data saved as:\n - Pickle: {savepath}.pkl\n - CSV: {csv_path}")


        return final_df


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("path", help="Path to get the train dataset")
    PARSER.add_argument("path_to_labels", help="Path to get the labels dataset")
    PARSER.add_argument("savepath", help="Path to save the pickle of the output dataframe")
    ARGS = PARSER.parse_args()
    LOAD_ARTICLES = CreateDataframeSLC.load_sentences_with_labels(ARGS.path,
                                                                  ARGS.path_to_labels,
                                                                  ARGS.savepath)
