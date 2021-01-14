'''
spam_filter.py
Spam v. Ham Classifier trained and deployable upon short
phone text messages.
'''
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

class SpamFilter:

    def __init__(self, text_train, labels_train):
        """
        Creates a new text-message SpamFilter trained on the given text 
        messages and their associated labels. Performs any necessary
        preprocessing before training the SpamFilter's Naive Bayes Classifier.
        As part of this process, trains and stores the CountVectorizer used
        in the feature extraction process.
        
        :param DataFrame text_train: Pandas DataFrame consisting of the
        sample rows of text messages
        :param DataFrame labels_train: Pandas DataFrame consisting of the
        sample rows of labels pertaining to each text message
        """
        vectorizer = CountVectorizer(stop_words='english')
        features = vectorizer.fit_transform(text_train)
        self.vectorizer = vectorizer
        clf = MultinomialNB()
        clf.fit(features, labels_train)
        self.clf = clf.fit(features, labels_train)
        
    def classify (self, text_test):
        """
        Takes as input a list of raw text-messages, uses the SpamFilter's
        vectorizer to convert these into the known bag of words, and then
        returns a list of classifications, one for each input text
        
        :param list/DataFrame text_test: A list of text-messages (strings) consisting
        of the messages the SpamFilter must classify as spam or ham
        :return: A list of classifications, one for each input text message
        where index in the output classes corresponds to index of the input text.
        """
        vector = self.vectorizer.transform(text_test)
        classifications = self.clf.predict(vector)
        return classifications
    
    def test_model (self, text_test, labels_test):
        """
        Takes the test-set as input (2 DataFrames consisting of test texts
        and their associated labels), classifies each text, and then prints
        the classification_report on the expected vs. given labels.
        
        :param DataFrame text_test: Pandas DataFrame consisting of the
        test rows of text messages
        :param DataFrame labels_test: Pandas DataFrame consisting of the
        test rows of labels pertaining to each text message
        """
        classes_predicted = self.classify(text_test)
        results = classification_report(labels_test, classes_predicted)
        return results
        
def load_and_sanitize (data_file):
    """
    Takes a path to the raw data file (a csv spreadsheet) and
    creates a new Pandas DataFrame from it with only the message
    texts and labels as the remaining columns.
    
    :param string data_file: String path to the data file csv to
    load-from and fashion a DataFrame from
    :return: The sanitized Pandas DataFrame containing the texts
    and labels
    """
    data = pd.read_csv(data_file, encoding="latin-1")
    data = data.rename(columns={"v1": "class", "v2": "text"},)
    data = data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
    return data

if __name__ == "__main__":
    data = load_and_sanitize('../dat/texts.csv')
    text_list = data['text'].tolist()
    class_list = data['class'].tolist()
    text_train, text_test, class_train, class_test = train_test_split(text_list, class_list, test_size=0.33, random_state=42)
    spam_filter = SpamFilter(text_train, class_train)
    test_spam = spam_filter.test_model(text_test, class_test)
    print("classification report: ", test_spam)