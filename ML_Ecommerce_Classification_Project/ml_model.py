# -*- coding: utf-8 -*-
"""ML_Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VC3mcqcfvS1HCEg3Db2GtloiNN0hEoIN

# Features Engineering

Read in the data
"""

import pandas as pd
import numpy as np
from datetime import datetime

## Read data from google drive 
from google.colab import drive
drive.mount('/content/drive/')
path = "/content/drive/Shareddrives/DDR&ML/"

df = pd.read_csv(f"{path}df_sample.csv")

"""Create and prepare a dataset df_targets to compile all variables that are used for modeling."""

df_targets = df.loc[df["event_type"].isin(["cart","purchase"])].drop_duplicates(subset=['event_type', 'product_id','price', 'user_id','user_session'])
df_targets["is_purchased"] = np.where(df_targets["event_type"]=="purchase",1,0)
df_targets["is_purchased"] = df_targets.groupby(["user_session","product_id"])["is_purchased"].transform("max")
df_targets = df_targets.loc[df_targets["event_type"]=="cart"].drop_duplicates(["user_session","product_id","is_purchased"])
df_targets['event_weekday'] = df_targets['event_time'].apply(lambda s: str(datetime.strptime(str(s)[0:10], "%Y-%m-%d").weekday()))
df_targets.dropna(how='any', inplace=True)
df_targets["category_code_level1"] = df_targets["category_code"].str.split(".",expand=True)[0].astype('category')
df_targets["category_code_level2"] = df_targets["category_code"].str.split(".",expand=True)[1].astype('category')

cart_purchase_users = df.loc[df["event_type"].isin(["cart","purchase"])].drop_duplicates(subset=['user_id'])
cart_purchase_users.dropna(how='any', inplace=True)
cart_purchase_users_all_activity = df.loc[df['user_id'].isin(cart_purchase_users['user_id'])]

activity_in_session = cart_purchase_users_all_activity.groupby(['user_session'])['event_type'].count().reset_index()
activity_in_session = activity_in_session.rename(columns={"event_type": "activity_count"})

df_targets = df_targets.merge(activity_in_session, on='user_session', how='left')
df_targets['activity_count'] = df_targets['activity_count'].fillna(0)

df_targets.head()

df_targets.info()

"""# Model Building"""

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from sklearn.utils import resample
from sklearn import metrics

is_purcahase_set = df_targets[df_targets['is_purchased']== 1]
is_purcahase_set.shape[0]

not_purcahase_set = df_targets[df_targets['is_purchased']== 0]
not_purcahase_set.shape[0]

# Resampling data because purchased & non-purchased are not evenly distributed 
n_samples = 2697
is_purchase_downsampled = resample(is_purcahase_set, replace = False, n_samples = n_samples, random_state = 42)
not_purcahase_set_downsampled = resample(not_purcahase_set,replace = False,n_samples = n_samples,random_state = 27)

downsampled = pd.concat([is_purchase_downsampled, not_purcahase_set_downsampled])
downsampled['is_purchased'].value_counts()

features = downsampled.loc[:,['brand', 'price', 'event_weekday', 'category_code_level1', 'category_code_level2', 'activity_count']]

"""Encode categorical variables"""

features.loc[:,'brand'] = LabelEncoder().fit_transform(downsampled.loc[:,'brand'].copy())
features.loc[:,'event_weekday'] = LabelEncoder().fit_transform(downsampled.loc[:,'event_weekday'].copy())
features.loc[:,'category_code_level1'] = LabelEncoder().fit_transform(downsampled.loc[:,'category_code_level1'].copy())
features.loc[:,'category_code_level2'] = LabelEncoder().fit_transform(downsampled.loc[:,'category_code_level2'].copy())

is_purchased = LabelEncoder().fit_transform(downsampled['is_purchased'])
features.head()

print(list(features.columns))

"""Split Data"""

X_train, X_test, y_train, y_test = train_test_split(features,is_purchased, test_size = 0.2,  random_state = 42)

"""# Model Building

## 1. Logistic Regression
"""

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Fit Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

# Print the evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("fbeta:", metrics.fbeta_score(y_test, y_pred, average='weighted', beta=0.5))

"""Importances of Variables """

importance = logreg.coef_[0]
feature_names = X_train.columns.tolist()
feature_importances = dict(zip(feature_names, importance))
sorted_features = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)
b
plt.barh(range(len(sorted_features)), [abs(x[1]) for x in sorted_features], align='center')
plt.yticks(range(len(sorted_features)), [x[0] for x in sorted_features])
plt.xlabel('Importance')
plt.title('Feature Importances (Logistic Regression)')
plt.show()

importance = logreg.coef_[0]
feature_names = X_train.columns.tolist()
feature_importances = dict(zip(feature_names, importance))
sorted_features = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)

print("Feature Importances:")
for feature, importance in sorted_features:
    print("%s: %.5f" % (feature, importance))

"""## 2. Random Forest"""

from sklearn.ensemble import RandomForestClassifier
# Create a random forest classifier object
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Predict the labels of the test data using the trained model
y_pred = rfc.predict(X_test)

# Print the evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("fbeta:", metrics.fbeta_score(y_test, y_pred, average='weighted', beta=0.5))

"""Importances of Variables"""

import matplotlib.pyplot as plt

# Get the feature importances from the trained model
importances = rfc.feature_importances_

# Get the column names from the training data
feature_names = X_train.columns

# Sort the features by importance in descending order
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))

# Plot the feature importances
plt.figure()
plt.title("Feature importances (Random Forest)")
plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

"""## 3. Neural Network"""

from keras import backend as K
from tensorflow import keras

nn_model = keras.Sequential([
    keras.layers.Dense(units=256, activation='relu',input_shape=(6,)),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid'),
])
# Define the evaluating metric function
def recall_func(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_func(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_func(y_true, y_pred):
    precision = precision_func(y_true, y_pred)
    recall = recall_func(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

nn_model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy',recall_func,precision_func,f1_func])
# fit neural network model
nn_model.fit(X_train, y_train, epochs=25)

loss,accuracy,recall,precision,f1_score = nn_model.evaluate(X_test, y_test, verbose=2)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("fbeta:", f1_score)