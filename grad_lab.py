"""
Graduation Lab: Week 6


Instructions:

Let's build a kNN model using the college completion data. 
The data is messy and you have a degrees of freedom problem, as in, we have too many features.  

You've done most of the hard work already, so you should be ready to move forward with building your model. 

1. Use the question/target variable you submitted and 
build a model to answer the question you created for this dataset (make sure it is a classification problem, convert if necessary). 

2. Build a kNN model to predict your target variable using 3 nearest neighbors. Make sure it is a classification problem, meaning
if needed changed the target variable.

3. Create a dataframe that includes the test target values, test predicted values, 
and test probabilities of the positive class.

4. No code question: If you adjusted the k hyperparameter what do you think would
happen to the threshold function? Would the confusion matrix look the same at the same threshold 
levels or not? Why or why not?

5. Evaluate the results using the confusion matrix. Then "walk" through your question, summarize what 
concerns or positive elements do you have about the model as it relates to your question? 

6. Create two functions: One that cleans the data & splits into training|test and one that 
allows you to train and test the model with different k and threshold values, then use them to 
optimize your model (test your model with several k and threshold combinations). Try not to use variable names 
in the functions, but if you need to that's fine. (If you can't get the k function and threshold function to work in one
function just run them separately.) 

7. How well does the model perform? Did the interaction of the adjusted thresholds and k values help the model? Why or why not? 

8. Choose another variable as the target in the dataset and create another kNN model using the two functions you created in
step 7. 

"""
#%%
# example of how I cleaned the data
# README for the dataset - https://data.world/databeats/college-completion/workspace/file?filename=README.txt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

grad_data = pd.read_csv('https://query.data.world/s/qpi2ltkz23yp2fcaz4jmlrskjx5qnp', encoding="cp1252")
# the encoding part here is important to properly read the data! It doesn't apply to ALL csv files read from the web,
# but it was necessary here.
grad_data.info()


#%%
# We have a lot of data! A lot of these have many missing values or are otherwise not useful.
to_drop = list(range(39, 56))
to_drop.extend([27, 9, 10, 11, 28, 36, 60, 56])
#%%
grad_data1 = grad_data.drop(grad_data.columns[to_drop], axis=1)
grad_data1.info()
#%%
# drop even more data that doesn't look predictive
drop_more = [0,2,3,6,8,11,12,14,15,18,21,23,29,32,33,34,35]
grad_data2 = grad_data1.drop(grad_data1.columns[drop_more], axis=1)
grad_data2.info()
#%%
print(grad_data2.head())
#%%
import numpy as np
grad_data2.replace('NULL', np.nan, inplace=True)
#%%
grad_data2['hbcu'] = [1 if grad_data2['hbcu'][i]=='X' else 0 for i in range(len(grad_data2['hbcu']))]
grad_data2['hbcu'].value_counts()
#%%
grad_data2['hbcu'] = grad_data2.hbcu.astype('category')
# convert more variables to factors
grad_data2[['level', 'control']] = grad_data2[['level', 'control']].astype('category')
#%%
# In R, we convert vals to numbers, but they already are in this import
grad_data2.info()
#%%
# check missing data
import seaborn as sns

sns.displot(
    data=grad_data2.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.25
)
#%%
#let's drop med_stat_value then delete the rest of the NA rows
grad_data2 = grad_data2.drop(grad_data[['med_sat_value']], axis=1)
grad_data2.dropna(axis = 0, how = 'any', inplace = True)
#%%
sns.displot(
    data=grad_data2.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.25
)

#%%
grad_data2.info()
grad_data2.head()
grad_data2["grad_100_value"].value_counts()








#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------

# The Question that I'll be answering with this data is if we can predict if a school is a 4 year or 2 year institution based on the other features in the dataset. 
# Those other features will be awards_per_value and grad_100_values



#%%
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier




# First I just want to visualize if there is a relationship between awards_per_value and grad_100_value based on level of institution

# %%
sns.scatterplot(data=grad_data2, x='awards_per_value', y='grad_100_value', hue='level')




# %%
# Standardize the data
def scale_numeric_columns_minmax(df):
	# scale all number columns to 0-1
	df = df.copy()
	numeric_cols = list(df.select_dtypes("number"))
	df[numeric_cols] = MinMaxScaler().fit_transform(df[numeric_cols])
	return df

grad_data_std = scale_numeric_columns_minmax(grad_data2)


# %%
# Define features and target variable
feature_cols = ["awards_per_value", "grad_100_value"]
X = grad_data_std[feature_cols]
y = grad_data_std["level"].astype(str)


# %%
# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


#%% 
# Build and fit the kNN model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

y_hat = knn_model.predict(X_test)

# make a dataframe that includes the test target values, test predicted values, and test probabilities of the positive class.
class_labels = list(knn_model.classes_)
positive_label = "4-year" if "4-year" in class_labels else class_labels[1]
positive_index = class_labels.index(positive_label)
y_proba_positive = knn_model.predict_proba(X_test)[:, positive_index]

knn_results = pd.DataFrame({
    "level_actual": y_test.values,
    "level_pred": y_hat,
    "proba_positive": y_proba_positive,
})

knn_results.head()

# %%
# Evaluate the results using the confusion matrix.
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(knn_results["level_actual"], knn_results["level_pred"], labels=knn_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()



'''

# 4.No code question: If you adjusted the k hyperparameter what do you think would
happen to the threshold function? Would the confusion matrix look the same at the same threshold 
levels or not? Why or why not?

Adjusting the K hyperparameter changes the number of neighbors considered when making predictions. 
A smaller K value (like 1 or 3) makes the model more sensitive to local patterns in the data, which can lead to overfitting. 
A larger K value smooths out predictions by considering more neighbors, which can help reduce noise but may also overlook local variations. 
So if we change K, the decision boundary of the model shifts, which in turn affects the predicted probabilities for each class.
This means that the confusion matrix would likely look different at the same threshold levels because the predicted class labels would change based on the new probabilities.

'''