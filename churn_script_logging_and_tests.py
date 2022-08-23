"""
A module to create logging and tests

Author: George Dialektakis
Date: August 2022
"""

import os
import logging
from churn_library import import_data, perform_eda

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# import churn_library_solution as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    """
        test data import - this example is completed for you to assist with the other test functions
    """
    try:
        data_df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data_df.shape[0] > 0
        assert data_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    """
        test perform eda function
    """
    try:
        perform_eda(import_data("./data/bank_data.csv"))

        # path of the directory of eda images
        path = "./images/eda"
        # Getting the list of directories
        dir = os.listdir(path)

        # Checking if the list is empty or not
        assert len(dir) != 0
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: The folder images/eda is empty which means eda images haven't been saved properly.")
        raise err

def test_encoder_helper(encoder_helper):
    """
        test encoder helper
        """


def test_perform_feature_engineering(perform_feature_engineering):
    """
        test perform_feature_engineering
        """


def test_train_models(train_models):
    """
        test train_models
        """


if __name__ == "__main__":
    test_import(import_data)
    test_eda(perform_eda)

