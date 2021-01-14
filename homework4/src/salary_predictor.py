'''
salary_predictor.py
Predictor of salary from old census data -- riveting!
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

class SalaryPredictor:
    
    def __init__(self, X_train, y_train):
        """
        Creates a new SalaryPredictor trained on the given features from the
        preprocessed census data to predicted salary labels. Performs and fits
        any preprocessing methods (e.g., imputing of missing features,
        discretization of continuous variables, etc.) on the inputs, and saves
        these as attributes to later transform test inputs.
        
        :param DataFrame X_train: Pandas DataFrame consisting of the
        sample rows of attributes pertaining to each individual
        :param DataFrame y_train: Pandas DataFrame consisting of the
        sample rows of labels pertaining to each person's salary
        """

        self.one_hot_encoder = preprocessing.OneHotEncoder(handle_unknown = 'ignore') 
        filtered_X = X_train.filter(items=['work_class', 'education', 'marital', 'occupation_code', 'relationship', 'race', 'sex', 'country'])

        self.one_hot_encoder.fit(filtered_X)
        self.feature_names = self.one_hot_encoder.get_feature_names()
        # print("train labels:")
        # print(self.feature_names)
        test = self.one_hot_encoder.transform(filtered_X).toarray()
        encoded_arr = self.encode(test, X_train) 
               
        self.clf = LogisticRegression(max_iter=100000).fit(encoded_arr, y_train)
    
    # encoded_arr: string vals that have been one hot encoded
    def encode(self, encoded_arr, X_file):
        age_list = X_file['age'].tolist()
        education_list = X_file['education_years'].tolist()
        capital_gain_list = X_file['capital_gain'].tolist()
        capital_loss_list = X_file['capital_loss'].tolist()
        hours_per_week_list = X_file['hours_per_week'].tolist()
        index = 0
        to_return = []
        for line in encoded_arr:
            to_add = [age_list[index], education_list[index], capital_gain_list[index],capital_loss_list[index], hours_per_week_list[index]]
            to_return.append(np.concatenate((line, to_add)))
            index = index + 1
    
        return to_return

    def classify (self, X_test):
        """
        Takes a DataFrame of rows of input attributes of census demographic
        and provides a classification for each. Note: must perform the same
        data transformations on these test rows as was done during training!
        
        :param DataFrame X_test: DataFrame of rows consisting of demographic
        attributes to be classified
        :return: A list of classifications, one for each input row X=x
        """
        filtered_X = X_test.filter(items=['work_class', 'education', 'marital', 'occupation_code', 'relationship', 'race', 'sex', 'country'])
        self.one_hot_encoder.fit(filtered_X)
        new_labels = self.one_hot_encoder.get_feature_names()
        # print("new labels:")
        # print(new_labels)

        test = self.one_hot_encoder.transform(filtered_X).toarray()
        encoded_arr = self.encode(test, X_test) 
        classifications = self.clf.predict(encoded_arr)
        return classifications
    
    def test_model (self, X_test, y_test):
        """
        Takes the test-set as input (2 DataFrames consisting of test demographics
        and their associated labels), classifies each, and then prints
        the classification_report on the expected vs. given labels.
        
        :param DataFrame X_test: Pandas DataFrame consisting of the
        sample rows of attributes pertaining to each individual
        :param DataFrame y_test: Pandas DataFrame consisting of the
        sample rows of labels pertaining to each person's salary
        """
        classes_predicted = self.classify(X_test)
        results = classification_report(y_test, classes_predicted)
        return results
    
        
def load_and_sanitize (data_file):
    """
    Takes a path to the raw data file (a csv spreadsheet) and
    creates a new Pandas DataFrame from it with the sanitized
    data (e.g., removing leading / trailing spaces).
    NOTE: This should *not* do the preprocessing like turning continuous
    variables into discrete ones, or performing imputation -- those
    functions are handled in the SalaryPredictor constructor, and are
    used to preprocess all incoming test data as well.
    
    :param string data_file: String path to the data file csv to
    load-from and fashion a DataFrame from
    :return: The sanitized Pandas DataFrame containing the demographic
    information and labels. It is assumed that for n columns, the first
    n-1 are the inputs X and the nth column are the labels y
    """
    # remove whitespace with regex delimiter
    data = pd.read_csv(data_file, encoding="latin-1", delimiter=' *, *', engine='python')
    data.replace('?', np.NaN, inplace=True)
    imp = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
    imp = imp.fit_transform(data)
    data = pd.DataFrame(data=imp, index=data.index, columns=data.columns)

    return data


if __name__ == "__main__":
    data = load_and_sanitize('../dat/salary.csv')
    info_list = data.drop('class', axis='columns')
    class_list = data['class'].tolist()
    info_train, info_test, class_train, class_test = train_test_split(info_list, class_list, test_size=0.33, random_state=42)
    predictor = SalaryPredictor(info_train, class_train)
    test = predictor.test_model(info_test, class_test)
    print("classification report: ", test)