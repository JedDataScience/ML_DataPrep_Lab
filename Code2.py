# %%
# Imports - Libraries needed for data manipulation and ML preprocessing
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
# Make sure to install sklearn in your terminal first!
# Use: pip install scikit-learn
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data


#  %%
# Load the second dataset 
# Got the data from the repo in class
placement = pd.read_csv("https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv")# %%

# %%
#Lets look at the data and get some basic info
placement.info()


#Question that can be answered with this data:
# TO what effect does the percentage of degree_P have on the salary of a student after graduation?
# The independent business metric for my problem is the accuracy of predicting whether a student will have a high or low salary after graduation based on their percentage of degree_P and other features.


# %%
#Change the columns to categorical variables if needed. 
#Chose these columns based on the values they take on and if they are categorical in nature. I also used Data Wrangler to help me identify categorical columns.
categorical_cols = [
	'gender',
	'ssc_b',
	'hsc_b',
	'hsc_s',
	'degree_t',
	'workex',
	'specialisation',
	'status'
]
placement[categorical_cols] = placement[categorical_cols].astype('category')
placement.dtypes
# I dont need to look at the value counts of these categorical variables because they all have a reasonable number of values in each category and dont need any collapsing of categories.
# Based on the value counts I saw in Data Wrangler.


# %% 
# Now I want to normalize the numerical columns using a minmax scaler
# To make this easier I will create a list of numeric values
abc = list(placement.select_dtypes('number')) #select function to find the numeric variables and create a list  
placement[abc] = MinMaxScaler().fit_transform(placement[abc])
placement #notice the difference in the range of values for the numeric variables


# %%
# I also want to do one hot encoding for the categorical variables so that they can be used in ML models
# This will expand the columns of my dataframe and create binary columns for each category in the categorical variables
placement = pd.get_dummies(placement, columns=categorical_cols)
placement 


# %%
# To get the prevelance of the target variable I will look at the value counts of the degree_p variable which is the percentage of degree_P that a student has. 
placement.boxplot(column='degree_p', vert=False, grid=False)
placement.degree_p.describe()
# I notice that the upper quartile of values is 0.53 which means that 75% of the students have a percentage of degree_P that is less than or equal to 0.53.

# %%
placement['degree_p_f'] = pd.cut(placement.degree_p, bins = [-1,0.537,1], labels =[0,1])
placement_clean = placement.dropna(subset=['degree_p_f'])
placement_clean
# I created a new binary variable called degree_p_f that categorizes the percentage of degree_P into two groups: 0 for students with a percentage of degree_P of 53.7% or lower, and 1 for students with a percentage of degree_P above 53.7%.

# %%
# Now Ill calculate the prevalence of students with a percentage of degree_P above 53.7%
prevalence = placement.degree_p_f.value_counts()[1]/len(placement.degree_p_f)
print(f"The prevalence of students with a percentage of degree_P above 53.7% is: {prevalence:.2f}")


# %%
# I will now partition the data using the prevalence of the target variable to create a training and testing set. 
# I will use the train_test_split function from sklearn and stratify by the degree_p_f variable to ensure that the distribution of the target variable is similar in both sets.

Train, Test = train_test_split(placement_clean, train_size=171, stratify=placement_clean.degree_p_f)

# %%
print(Train.shape)
print(Test.shape)

# %%
# Now we can use it again to create a tunning set.
Tune, Test = train_test_split(Test,  train_size = .20, stratify= Test.degree_p_f)
print(Tune.shape)
print(Test.shape)

# %%
