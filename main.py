import numpy as np
import pandas   as pd
import matplotlib.pyplot as plt # The matplotlib library is a library used for creating graphs and visualizations in Python.
import seaborn as sns 

"""
This code imports the seaborn library, which is a library for statistical data visualization in Python. 
It provides a high-level interface for drawing attractive and informative statistical graphics.
"""

import plotly.express as px

"""The code imports the plotly.express library as px. This library provides a simple way to create interactive, 
    web-based plots and charts with Python. It is built on top of the Plotly.js library and can be used to create many different 
    types of plots such as line charts, bar charts, scatter plots, bubble charts, histograms, and more.
"""
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

"""
This code imports three different metrics from the Scikit-Learn library for machine learning in Python: accuracy_score, 
    confusion_matrix, and classification_report. These metrics can be used to evaluate the performance of a machine learning model, 
    such as a classification algorithm. The accuracy_score metric calculates the percentage of correct predictions made by the model, 
    while the confusion_matrix and classification_report metrics provide more detailed analysis of the model's performance.
"""

from yellowbrick.classifier import ConfusionMatrix
"""This code imports the ConfusionMatrix class from the yellowbrick library. The ConfusionMatrix class is used to visualize confusion matrices 
for classification tasks in order to better understand the performance of a classifier."""

from sklearn.tree import DecisionTreeClassifier
"""This code imports the DecisionTreeClassifier class from the sklearn.tree module. The DecisionTreeClassifier is a supervised machine learning 
algorithm that can be used to classify data points into one of several different classes. It works by building a decision tree from the training 
data and then using it to make predictions on unseen data points."""

from sklearn.metrics import r2_score
"""
This code imports the r2_score module from the scikit-learn library. This module is used for calculating the coefficient of determination (R-squared) 
for a given model. R-squared is a measure of how well the model fits the data, with a value of 1 indicating a perfect fit."""

from sklearn.metrics import mean_absolute_error, mean_squared_error
"""This code imports the mean_absolute_error and mean_squared_error functions from the sklearn.metrics library. These functions are used to evaluate the 
performance of a machine learning model by calculating the difference between predicted values and the actual values."""

from sklearn.model_selection import GridSearchCV
"""GridSearchCV is a scikit-learn module that is used to perform hyperparameter tuning to optimize the performance of a machine learning model. It exhaustively 
searches over a given set of hyperparameters to find the best combination that gives the highest performance. The GridSearchCV object takes an estimator (usually a classifier) 
and a set of parameters to search over as inputs, and it trains and evaluates the model for each combination of parameters using cross-validation."""

df = pd.read_csv('C:/Users/Carnot/Documents/Loan-Eligibility---EDA-Balancing-and-ML/Loan_Data.csv', sep = ',')
pd.set_option('display.max_colums', None)
"""
This code reads in a csv file located at the specified path, separates the data by commas, and sets the maximum number of columns that can be displayed to none."""

pd.head()

