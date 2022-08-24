# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project, I implement my learnings from Clean Code Principles Course from ML DevOps Udacity Nanodegree to identify credit card customers that are most likely to churn. 
This is a project to implement best coding practices and the code for this project is not developed entirely from scratch.

More precisely I implement the code available in the churn_notebook.ipynb file containing the solution to identify credit card customers that are most likely to churn, but following the engineering and software best practices.

## Files and data description
Overview of the files and data present in the root directory. 
- `churn_library.py` is a library of functions to find customers who are likely to churn but written following clean code principles.
- `churn_script_logging_and_tests.py` contains unit tests for the churn_library.py functions. The goal of test functions is to checking the returned items aren't empty or folders where results should land have results after the function has been run. In addition, it logs info messages and errors in a `churn_library.log` file inside `logs` folder.
- `churn_notebook.ipynb` is the given notebook with the code that we need to make more efficient.
- `data` is a folder containing our data in csv format.
- `images` is a folder with two subfolders: `eda` and `results` containing plots from the exploratory data analysis and the results after training and evaluating the ML models, respectively.
- `models` stores the trained ML models in pickle format for later loading.
## Running Files
To run the code, first clone the latest version of the project to any directory of your choice:
```
git clone https://github.com/gdialektakis/Predict-Customer-Churn-ML-DevOps-Udacity-Nanodegree.git
```

Install dependencies:

for python 3.6:
```
pip install -r requirements_py3.6.txt
```
for python 3.8
```
pip install -r requirements_py3.8.txt
```


### Running the code:
To run the script for detecting customer churn:
```
ipython churn_library.py
```

To run tests associated with churn_library.py functions you can run:
```
ipython churn_script_logging_and_tests.py
```
Then check `logs/churn_library.log` to find if there were any errors for any of the tests.