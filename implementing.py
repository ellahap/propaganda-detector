import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from slc_analysis import SLC
from sklearn.metrics import accuracy_score, classification_report
import os


class ImplementHQP():
    @staticmethod
    def cleanNew():
        df = pd.read_csv("df_tweets_HiQualProp.csv")
        # print(df.isna().sum()) # no n/a values
        # print(df.dtypes) # looks good
        print(df.shape)
        print(df.head())
        return df

    @staticmethod
    def annotate():
        new_df = ImplementHQP.cleanNew()

        new_df["is_prop"] = (new_df.propaganda_category != "none")

        print(new_df.head())

        # Load spaCy for sentence segmentation
        nlp = spacy.load("en_core_web_sm")
        vectorizer, clf = SLC.buildModel("LogReg")  # Load TF-IDF and Logistic Regression

        results = []
        for _, row in new_df.iterrows():
            annotated_prop = row["is_prop"]
            text = row["text_normalized"]
            sentences = [sent.text.strip() for sent in nlp(text).sents]
            sentence_preds = clf.predict(vectorizer.transform(sentences))
            pred_prop = int(any(sentence_preds))
            results.append((text, sentences, sentence_preds, annotated_prop, pred_prop))


        df_annotated = pd.DataFrame(results, columns=["Text", "Sentences", "Sentence_Labels", "annotated_prop", "pred_prop"])


        print(df_annotated.head())


        accuracy = accuracy_score(df_annotated["annotated_prop"], df_annotated["pred_prop"])
        report = classification_report(df_annotated["annotated_prop"], df_annotated["pred_prop"])
        
        print("Accuracy:", accuracy)
        print("Classification Report:\n", report)
        
        df_annotated.to_csv("annotated_df_tweets.csv", index=False)
        print("Annotation complete. Annotated file saved as 'annotated_df_tweets.csv'.")






class ImplementNews():
    @staticmethod
    def cleanNew():
        df = pd.read_csv("NewsArticles.csv", encoding='latin1')
        # print(df.isna().sum()) # no n/a values
        # print(df.dtypes) # looks good
        cropped_df = df[["article_id", "publish_date", "article_source_link", "title", "subtitle", "text"]]
        cropped_df.dropna(subset=['text'], inplace=True)
        print(cropped_df.shape)
        print(cropped_df.head())
        return cropped_df
    
    @staticmethod
    def addCleanSource(df):
        df["Source"] = df["article_source_link"].apply(
            lambda x: "ABC" if "abcnews.go" in x 
                else "BBC" if "bbc.co" in x 
                    else "CNN" if "cnn.com" in x 
                        else "Deutsche Welle" if "dw.com" in x 
                            else "TASS" if "tass.com" in x 
                                else "Huffington Post" if "huffingtonpost.com" in x 
                                    else "Al Jazeera" if "aljazeera.com" in x 
                                        else "China Daily" if "chinadaily.com" in x 
                                            else "RTE" if "rte.ie" in x 
                                                else "NAN"
        )
        return df
    
    


    @staticmethod
    def createGraphs(df):
        # Bar Plot, Sources vs. Proportion
        plt.figure(figsize=(14, 6))
        plt.bar(df['Source'], df['Proportion Propaganda'], color='skyblue')

        plt.xlabel('Sources')
        plt.ylabel('Proportion Propaganda')
        plt.title('Proportion of Articles Containing Propaganda by News Source')

        plt.savefig("./graphs/bar-chart.png", dpi=300) # save to graphs folder
        plt.close()


        # Scatter Plot
        df_scatter = df[df["Media Bias"] != -99]

        print(df_scatter['Media Bias'])

        # Get unique sources and assign each a distinct color using a colormap
        unique_sources = df_scatter['Source'].unique()
        colors = plt.cm.get_cmap('tab10', len(unique_sources))
        
        # Loop over each source, plotting its points with a unique color and label
        for i, source in enumerate(unique_sources):
            group = df_scatter[df_scatter['Source'] == source]
            plt.scatter(group['Media Bias'], group['Proportion Propaganda'], 
                        label=source, color=colors(i))

        # Add labels and title
        plt.xlabel('Media Bias')
        plt.ylabel('Proportion Propaganda')
        plt.title('Media Bias vs. Proportion of Articles Containing Propaganda')

        plt.legend(title='Source')

        # Show the plot
        plt.savefig("./graphs/scatter-plot.png", dpi=300) # save to graphs folder
        plt.close()

        



    @staticmethod
    def annotate():
        # Load dataset
        new_df = ImplementNews.cleanNew()
        new_df = ImplementNews.addCleanSource(new_df)


        # Load spaCy for sentence segmentation
        nlp = spacy.load("en_core_web_sm")
        vectorizer, clf = SLC.buildModel("LogReg")  # Load TF-IDF and Logistic Regression


        results = []
        for _, row in new_df.iterrows():
            source = row["Source"]
            text = row["text"]
            sentences = [sent.text.strip() for sent in nlp(text).sents]
            sentence_preds = clf.predict(vectorizer.transform(sentences))
            is_prop = int(any(sentence_preds))
            results.append((source, text, sentences, sentence_preds, is_prop))


        df_annotated = pd.DataFrame(results, columns=["Source", "Text", "Sentences", "Sentence_Labels", "is_prop"])

        df_annotated.to_csv("annotated_news.csv", index=False)
        print("Annotation complete. File saved as 'annotated_news.csv'.")

        print(df_annotated.head())

        
        
        df_source_grouped_sum = df_annotated.groupby("Source")["is_prop"].sum().reset_index()
        df_source_grouped_count = df_annotated.groupby("Source")["is_prop"].count().reset_index()
        df_source_grouped_prop = pd.DataFrame({'Source': df_source_grouped_sum.Source, 'Proportion Propaganda': (df_source_grouped_sum.is_prop / df_source_grouped_count.is_prop)})

        df_source_grouped_prop["Media Bias"] = df_source_grouped_prop["Source"].apply(
            lambda x: -1.42 if "ABC" in x 
                else -0.80 if "BBC" in x 
                    else -1.30 if "CNN" in x 
                        else 0 if "Deutsche Welle" in x
                            else -99 if "TASS" in x # missing value
                                else -4.30 if "Huffington Post" in x 
                                    else -2.30 if "Al Jazeera" in x 
                                        else -99 if "China Daily" in x 
                                            else -99 if "RTE" in x 
                                                else -99
        )

        print(df_source_grouped_prop)
        ImplementNews.createGraphs(df_source_grouped_prop)


        


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    ImplementHQP.annotate()
    ImplementNews.annotate()
