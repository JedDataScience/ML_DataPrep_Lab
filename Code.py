
#%% 

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


# %%
## Load College Completion Data
# Data Source: UVa DS 3021 Course Repository
college_completions_url = "https://raw.githubusercontent.com/UVADS/DS-3021/main/data/cc_institution_details.csv"
college = pd.read_csv(college_completions_url)

## Question that can be answered with this data using classification or regression models:
# Can we predict the graduation rate of a college based on its characteristics such as size, control 
# The independent business metric for my problem is the accuracy of predicting whether a college has a high or low graduation rate based on its features.



# %%
college.info()
# %%

# %% 
#Select relevant features for analysis: I chose these columns based on the description of the data and what I thought would be relevant to the question I want to answer.
#The Columns that I selected dont have a lot of missing values so I decided to keep them
college = college.iloc[:, [3,4,5,6,7,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]]
college.info()



# %%
#Convert categorical columns to category dtype
college['level'] = college['level'].astype('category')
college['control'] = college['control'].astype('category')
college.dtypes


# %%
# I want to look at the categorical variables to see their possible values 
college['level'].value_counts()
# It only can take on two values: 4-year and 2-year so I dont need to do any collapsing of categories

# %%
college['control'].value_counts()
# This variable has three values: Public, Private Non-Profit, and Private For-Profit
# and they have enough values in each category to keep as is and have no need to collapse categories





# %%
# Now I will start to normalize the numerical columns using a minmax scaler
# To make this easier I will create a list of numeric values
abc = list(college.select_dtypes('number')) #select function to find the numeric variables and create a list  
college[abc] = MinMaxScaler().fit_transform(college[abc])
college #notice the difference in the range of values for the numeric variables


# %%
# Lets do the one hot encoding for the categorical variables
college = pd.get_dummies(college, columns=['level','control'])

# %%
# I want ot get the prevalence of the target variable, which is the graduation rate, to see if it is imbalanced
college.boxplot(column='grad_100_value', vert=False, grid=False)
college.grad_100_value.describe()
# I notice that the upper quartile of values is above 0.4365, which means that 75% of the colleges have a graduation rate below 43.65%. 
# This suggests that the target variable is somewhat imbalanced, with a majority of colleges having lower graduation rates. 
# This is important to keep in mind when building classification models, as it may affect the performance and require techniques to address the imbalance.


# %%
college['grad_100_value_f'] = pd.cut(college.grad_100_value, bins = [-1,0.4365,1], labels =[0,1])
college_clean = college.dropna(subset=['grad_100_value_f'])
college 
# I created a new binary variable called grad_100_value_f that categorizes the graduation rate into two groups: 0 for colleges with a graduation rate of 43.65% or lower, and 1 for colleges with a graduation rate above 43.65%. 
# This will allow me to use this variable as a target for classification models, where I can predict whether a college has a high or low graduation rate based on its characteristics.

# %%
#So now let's check the prevalence 
prevalence = college_clean.grad_100_value_f.value_counts()[1]/len(college_clean.grad_100_value_f)
print(f"The prevalence of colleges with a graduation rate above 43.65% is: {prevalence:.2f}")
# The prevalence of colleges with a graduation rate above 43.65% is: 0.25, which means that 25% of the colleges in the dataset have a graduation rate above 43.65%.


# %%
# I want to know partition the data into training and testing sets.
Train, Test = train_test_split(college_clean, train_size=2600, stratify=college_clean.grad_100_value_f)


# %%
print(Train.shape)
print(Test.shape)

# %%
# Now we can use it again to create a tunning set. 
Tune, Test = train_test_split(Test,  train_size = .25, stratify= Test.grad_100_value_f)
print(Tune.shape)
print(Test.shape)

# %%









##SECOND DATASET

# %%
# Load the second dataset 
