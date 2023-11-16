import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pickle



data0 = pd.read_csv('Datafiles/urldata.csv')
data0.head()

#Dropping the Domain column
data = data0.drop(['Domain'], axis = 1).copy()

#checking the data for null or missing values
data.isnull().sum()

# shuffling the rows in the dataset so that when splitting the train and test set are equally distributed
data = data.sample(frac=1).reset_index(drop=True)


# Sepratating & assigning features and target columns to X & y
y = data['Label']
X = data.drop('Label',axis=1)


# Splitting the dataset into train and test sets: 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)


# instantiate the model
xgb = XGBClassifier(learning_rate=0.4,max_depth=7)
#fit the model
xgb.fit(X_train, y_train)

#predicting the target value from the model for the samples
y_test_xgb = xgb.predict(X_test)
y_train_xgb = xgb.predict(X_train)

#computing the accuracy of the model performance
acc_train_xgb = accuracy_score(y_train,y_train_xgb)
acc_test_xgb = accuracy_score(y_test,y_test_xgb)

# save XGBoost model to file
pickle.dump(xgb, open("XGBoostClassifier.pickle.dat", "wb"))


# load model from file
loaded_model = pickle.load(open("XGBoostClassifier.pickle.dat", "rb"))