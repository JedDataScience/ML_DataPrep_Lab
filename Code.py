
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

## Question that can be answered with this data:
# what are the different types of institutions that have the highest completion rates and what variables are most correlated with completion rates?



# %%
college.info()
# %%

# %% 
#Select relevant features for analysis: I chose these columns based on their potential impact on college completion rat
#The Columns that I selected dont have a lot of missing values so I decided to keep them
columns_to_drop = [,]
college = college.iloc[:, [,]]
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
college.info()

# %%
