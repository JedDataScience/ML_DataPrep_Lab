
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

# %%
college.info

# %%
