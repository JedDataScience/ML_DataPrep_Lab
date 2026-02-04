import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_csv(url):
	# read a csv from a link
	return pd.read_csv(url)


def select_columns(df, column_indices):
	# keep only the columns we want
	return df.iloc[:, column_indices].copy()


def set_categorical_columns(df, columns):
	# tell pandas these columns are categories
	df = df.copy()
	df[columns] = df[columns].astype("category")
	return df


def scale_numeric_columns_minmax(df):
	# scale all number columns to 0-1
	df = df.copy()
	numeric_cols = list(df.select_dtypes("number"))
	df[numeric_cols] = MinMaxScaler().fit_transform(df[numeric_cols])
	return df


def one_hot_encode(df, columns):
	# turn categories into 0/1 columns
	return pd.get_dummies(df, columns=columns)


def add_binary_target_from_cut(df, source_col, target_col, bins, labels):
	# make a new 0/1 target based on ranges
	df = df.copy()
	df[target_col] = pd.cut(df[source_col], bins=bins, labels=labels)
	return df.dropna(subset=[target_col])


def split_train_test_tune(df, target_col, train_size, tune_size):
	# split into train, tune, and test with same target mix or in other works same prevalence of the target variable in each set. 
	train_df, test_df = train_test_split(
		df,
		train_size=train_size,
		stratify=df[target_col],
	)
	tune_df, test_df = train_test_split(
		test_df,
		train_size=tune_size,
		stratify=test_df[target_col],
	)
	return train_df, tune_df, test_df
