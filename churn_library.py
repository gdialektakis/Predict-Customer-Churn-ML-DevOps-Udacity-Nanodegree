"""
This module is a library of functions to find customers who are likely to churn.

Author: George Dialektakis
Date: August 2022
"""

# import libraries
import os

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_df: pandas dataframe
    """
    data_df = pd.read_csv(pth)
    print(data_df.head())
    return data_df


def perform_eda(data_df):
    """
    perform eda on data_df and save figures to images folder
    input:
            data_df: pandas dataframe

    output:
            None
    """
    print(data_df.shape)
    print(data_df.isnull().sum())
    print(data_df.describe())

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    quant_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio'
    ]

    data_df['Churn'] = data_df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    plt.figure(figsize=(20, 10))
    data_df['Churn'].hist()
    plt.savefig('./images/eda/churn_histogram.png')

    plt.figure(figsize=(20, 10))
    data_df['Customer_Age'].hist()
    plt.savefig('./images/eda/customer_age_histogram.png')

    plt.figure(figsize=(20, 10))
    data_df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/marital_status_histogram.png')

    plt.figure(figsize=(20, 10))
    # distplot is deprecated. Use histplot instead
    # sns.distplot(df['Total_Trans_Ct']);
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained using a kernel density estimate
    sns.histplot(data_df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('./images/eda/total_trans_histogram.png')

    plt.figure(figsize=(20, 10))
    sns.heatmap(data_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/correlation_heatmap.png')
    # plt.show()


def encoder_helper(data_df, category_lst, response=None):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """

    # encode columns that belong to category list
    for feature in category_lst:
        feature_lst = []
        feature_groups = data_df.groupby(feature).mean()['Churn']

        for val in data_df[feature]:
            feature_lst.append(feature_groups.loc[val])

        column_name = str(feature) + '_Churn'
        data_df[column_name] = feature_lst

    return data_df


def perform_feature_engineering(data_df, response):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """

    y = data_df['Churn']
    X = pd.DataFrame()


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    pass


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    pass


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass


if __name__ == "__main__":
    data_df = import_data("./data/bank_data.csv")
    perform_eda(data_df)
    category_lst = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category', 'Card_Category']
    encoder_helper(data_df, category_lst)
