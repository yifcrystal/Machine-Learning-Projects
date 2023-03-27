#!/usr/bin/env python
# coding: utf-8

# # Ensemble Study (Key on Random Forest)

# Reference :
# - https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/?
# - https://s3.amazonaws.com/kajabi-storefronts-production/file-uploads/sites/2147512189/themes/2150624317/downloads/4eb77-adc8-7d84-6ec8-8781264f6417_Random_Forest.pdf
# - https://s3.amazonaws.com/kajabi-storefronts-production/file-uploads/sites/2147512189/themes/2150624317/downloads/e4fbc2f-c755-ed1a-c18-f18ec25eb0d_Ensemble_Learning_Bagging_Boosting_and_Stacking.pdf
# - https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/
# - https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-HowItWorks.html

# **Contents** : 
# 
# Simple Ensemble Techniques 
# - [1. Max Voting](#mv)
# - [2. Averaging](#a)
# - [3. Weighted Averaging](#wa)
# 
# 
# Complex Ensemble Techniques
# - [1.Bagging](#bg)
#     - [Random Forest](#rf)
# - [2.Boosting](#bt)
#     - [AdaBoost](#ab)
#     - [Gradient Boosting (GBM)](#gbm) 
#     - [XGBoost](#xgb) 
#     - [Light GBM](#lgbm) 
# - [3.Stacking](#s)
# - [4.Blending](#bl)

# ## 1. Definition

# Ensemble learning is a machine learning technique where multiple models are trained on a dataset to make predictions, and the **predictions of those models are combined to produce a more accurate and robust prediction** than any of the individual models. In other words, ensemble learning is about combining the predictions of several weaker models to create a stronger model.

# ## 2. Types

# ### Simple Ensemble Techniques

# <a name="mv"></a>
# #### 1. Max Voting

# **Definition :** The max voting method is generally used for classification problems. In this technique, multiple models are used to make predictions for each data point. The predictions by each model are considered as a ‘vote’. **The predictions which we get from the majority of the models are used as the final prediction**.

# **Sample Code:**

# In[1]:


import pandas as pd
# read the text file into a pandas dataframe
df = pd.read_csv("/Users/crystal/Desktop/Random Forest/heart.csv")


# In[2]:


# IMPORTS
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statistics as st
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# SPLITTING THE DATASET
x = df.drop('target', axis = 1)
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[4]:


# MODELS CREATION
model1 = DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)


# The final prediction is stored in a numpy array called final_pred which is initialized as an empty array using the np.array([]) function. The code then runs a for loop from 0 to the length of the x_test variable (which represents the test dataset). Inside the loop, the mode function from the statistics library is used to calculate the mode of the three model predictions for each observation in the test dataset. These three model predictions are stored in pred1, pred2, and pred3. **The mode function returns the most common prediction value among the three predictions.** The resulting mode prediction is appended to the final_pred array using the np.append function. Finally, the final_pred array is printed using the print function to show the mode predictions for each observation in the test dataset.

# In[5]:


# PREDICTION
pred1=model1.predict(x_test)
pred2=model2.predict(x_test)
pred3=model3.predict(x_test)

# FINAL_PREDICTION
final_pred = np.array([])
for i in range(0,len(x_test)):
    final_pred = np.append(final_pred, st.mode([pred1[i], pred2[i], pred3[i]]))
print(final_pred)


# Alternatively, you can use **“VotingClassifier”** module in sklearn as follows:

# In[6]:


from sklearn.ensemble import VotingClassifier
model1 = LogisticRegression(random_state=1)
model2 = DecisionTreeClassifier(random_state=1)
model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')
model.fit(x_train,y_train)
model.score(x_test,y_test)


# <a name="a"></a>
# #### 2. Averaging

# **Definition:** : Similar to the max voting technique, multiple predictions are made for each data point in averaging. In this method, we take an average of predictions from all the models and use it to make the final prediction. **Averaging can be used for making predictions in regression problems or while calculating probabilities for classification problems.**

# In[7]:


model1 = DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

pred1=model1.predict_proba(x_test)
pred2=model2.predict_proba(x_test)
pred3=model3.predict_proba(x_test)

finalpred=(pred1+pred2+pred3)/3


# In[8]:


#print(finalpred)


# <a name="wa"></a>
# #### 3. Weighted Averaging

# **Definition:** This is an extension of the averaging method. All models are assigned different weights defining the importance of each model for prediction. 

# In[9]:


model1 = DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

pred1=model1.predict_proba(x_test)
pred2=model2.predict_proba(x_test)
pred3=model3.predict_proba(x_test)

finalpred=(pred1*0.3+pred2*0.3+pred3*0.4)


# In[10]:


#finalpred


# ### Complex Ensemble Techniques
# 
# <a name="bg"></a>
# #### 1. Bagging

# **Definition :**  <br>
# Bagging, short for **Bootstrap Aggregating**, is a machine learning technique that involves training multiple models (such as decision trees) on **different subsets of the original training data**, created through bootstrapping, which is a sampling technique that involves **creating subsets of observations from the original dataset with replacement**. The final output is determined by combining the outputs of all the models through a voting process. Bagging is often used to improve the accuracy and stability of machine learning models, especially in cases **where the training data is noisy or prone to overfitting**. Random Forest is an example of a machine learning algorithm that uses the bagging technique.

# **Models :** 
# - Bagging meta-estimator
# - Random forest

# <a name="rf"></a>
# #### Random Forest (Key)

# **1). Steps :**
# 1. Draw a random bootstrap sample of size (randomly choose examples from the training dataset with replacement).
# 2. Grow a decision tree from the bootstrap sample. At each node: 
# - Randomly select features without replacement. e.g., if there are 20 features, choose a random five as candidates for constructing the best split.
# - Split the node using the feature that provides the best split according to the objective function, e.g., maximizing the information gain.
# 3. Repeat steps 1-2 times. Essentially, we will build decision trees.
# 4. Aggregate the prediction by each tree to assign the class label by majority vote (classification) or take the average (regression).

# **2). Features**
# - Diversity: Not all attributes/variables/features are considered while making an individual tree; **each tree is different**.
# - Immune to the curse of dimensionality: **Since each tree does not consider all the features, the feature space is reduced**.
# - Parallelization: Each tree is created independently out of different data and attributes. This means we can fully use the CPU to build random forests.
# - Train-Test split: In a random forest, **we don’t have to segregate the data for train and test as there will always be 30% of the data which is not seen by the decision tree**.
# - Stability: Stability arises because the result is based on **majority voting/ averaging**.

# **3). Pros & Cons** <br>
# *Pros* : <br>
# 1. Has a better generalization performance than an individual decision tree due to randomness (the combination of bootstrap samples and using a subset of features), which helps to **decrease the model’s variance (thus low overfitting)**. So it corrects decision trees' habit of overfitting the training data.
# 2. Doesn’t require much parameter tuning. Using full-grown trees seldom costs much and results in fewer tuning parameters.
# 3. **Less sensitive to outliers in the dataset**.
# 4. It generates feature importance which is helpful when interpreting the results. <br>
# *Con* : <br>
# 1. Computationally expensive. It is fast to train but quite slow to create predictions once trained. More accurate models require more trees, which means using the model becomes slower.

# **4). Hyperparameters**
# 1. Hyperparameters to Increase the Predictive Power
# - n_estimators: Number of trees the algorithm builds before averaging the predictions.
# - max_features: Maximum number of features random forest considers splitting a node.
# - mini_sample_leaf: Determines the minimum number of leaves required to split an internal node.
# - criterion: How to split the node in each tree? (Entropy/Gini impurity/Log Loss)
# - max_leaf_nodes: Maximum leaf nodes in each tree
# 
# 2. Hyperparameters to Increase the Speed
# - n_jobs: it tells the engine how many processors it is allowed to use. If the value is 1, it can use only one processor, but if the value is -1, there is no limit.
# - random_state:controls randomness of the sample. The model will always produce the same results if it has a definite value of random state and has been given the same hyperparameters and training data.
# - oob_score: OOB means out of the bag. It is a random forest cross-validation method. In this, one-third of the sample is not used to train the data; instead used to evaluate its performance. These samples are called out-of-bag samples.

# **Sample Code:**

# In[11]:


# Importing the required libraries
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


import pandas as pd
df_02 = pd.read_csv('/Users/crystal/Desktop/Random Forest/heart_v2.csv', delimiter=',')


# In[13]:


df_02.head()


# In[14]:


sns.countplot(df_02['heart disease'])
plt.title('Value counts of heart disease patients')
plt.show()


# In[15]:


#3. Putting Feature Variable to X and Target variable to y.
# Putting feature variable to X
X = df_02.drop('heart disease',axis=1)
# Putting response variable to y
y = df_02['heart disease']


# In[16]:


#4. Train-Test-Split is performed
# now lets split the data into train and test
from sklearn.model_selection import train_test_split
# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
X_train.shape, X_test.shape


# In[17]:


#5. Let’s import RandomForestClassifier and fit the data.
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5,
                                       n_estimators=100, oob_score=True)


# In[18]:


get_ipython().run_cell_magic('time', '', 'classifier_rf.fit(X_train, y_train)')


# In[19]:


# checking the oob score
classifier_rf.oob_score_


# In[20]:


#6. Let’s do hyperparameter tuning for Random Forest using GridSearchCV and fit the data.
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
params = {
    'max_depth': [2,3,5,10,20],
    'min_samples_leaf': [5,10,20,50,100,200],
    'n_estimators': [10,25,30,50,100,200]
}
from sklearn.model_selection import GridSearchCV
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy")


# In[21]:


get_ipython().run_cell_magic('time', '', 'grid_search.fit(X_train, y_train)')


# In[22]:


grid_search.best_score_


# In[23]:


rf_best = grid_search.best_estimator_
rf_best


# From hyperparameter tuning, we can fetch the best estimator, as shown. The best set of parameters identified was max_depth=5, min_samples_leaf=10,n_estimators=10

# In[24]:


#7. Now, let’s visualize
from sklearn.tree import plot_tree
plt.figure(figsize=(80,40))
plot_tree(rf_best.estimators_[5], feature_names = X.columns,class_names=['Disease', "No Disease"],filled=True)


# In[25]:


from sklearn.tree import plot_tree
plt.figure(figsize=(80,40))
plot_tree(rf_best.estimators_[7], feature_names = X.columns,class_names=['Disease', "No Disease"],filled=True)


# In[26]:


#8. Now let’s sort the data with the help of feature importance
rf_best.feature_importances_


# In[27]:


imp_df = pd.DataFrame({
    "Varname": X_train.columns,
    "Imp": rf_best.feature_importances_
})
imp_df.sort_values(by="Imp", ascending=False)


# <a name="bt"></a>
# #### 2. Boosting 

# **Definition :** <br>
# Boosting is an ensemble learning technique in which multiple weak models are combined to generate a final output. **Unlike bagging, boosting works by building models in series, where each subsequent model attempts to correct the errors of the previous model**.<br>
# In boosting, each model is trained on the original data set, but with different weights assigned to each data point. *The weights of the misclassified data points are increased, while the weights of the correctly classified data points are decreased.* This helps the subsequent models to focus more on the misclassified data points and improve the overall accuracy of the final model.<br>
# On the other hand, bagging is also an ensemble learning technique in which **multiple models are trained independently on different subsets of the original data set**. In bagging, the subsets are generated through bootstrap sampling, which involves randomly selecting a subset of the data set with replacement. Each model generated from the bootstrap sample is trained independently, and the final output is obtained by combining the results of all models through majority voting. <br>
# In summary, the main difference between bagging and boosting is the way the models are combined. **Bagging involves training multiple models independently and combining their results through majority voting, while boosting involves building models in series, with each subsequent model attempting to correct the errors of the previous model**.

# <img style="float: left;" src="https://lh3.googleusercontent.com/bEfpUmjNGzKBCV6gEq6GzWeELTCYEoabucBughc-tUmkKA-j8eM04dBglRwaz4amaGS4ut3EbQJ3a_Nv9VA6sGNoGDonxv8mUg_ysN2goqu0WlIP38hvm7w2QSs5MBcNWwqK2xw5=s0" width="70%"> 

# **Models :**
# - AdaBoost
# - GBM
# - XGBM
# - Light GBM
# - CatBoost

# **Difference between Bagging and Boosting**

# <img style="float: left;" src="https://av-eks-blogoptimized.s3.amazonaws.com/4661536426211ba43ea612c8e1a6a1ed4550721164.png" width="35%"> 

# |  | Individual Learners | Bias-Variance |
# | :-----| ----: | :----: |
# | **Bagging** | Independent, Built in parallel | Reduce variance |
# | **Boosting** | Dependent, Built sequentially | Reduce bias |

# **Steps:**

# 1. A subset is created from the original dataset.
# 2. Initially, all data points are given equal weights.
# 3. A base model is created on this subset.
# 4. This model is used to make predictions on the whole dataset. <br>
# <img style="float: left;" src="https://av-eks-blogoptimized.s3.amazonaws.com/dd1-e1526989432375.png" width="15%">

# 5. Errors are calculated using the actual values and predicted values.
# 6. The observations which are incorrectly predicted, are given higher weights.*(Here, the three misclassified blue-plus points will be given higher weights)*
# 7. Another model is created and predictions are made on the dataset.
# (This model tries to correct the errors from the previous model) <br>
# <img style="float: left;" src="https://av-eks-blogoptimized.s3.amazonaws.com/dd2-e1526989487878.png" width="15%"> 

# 8. Similarly, multiple models are created, each correcting the errors of the previous model.
# 9. The final model (strong learner) is the weighted mean of all the models (weak learners) <br>
# <img style="float: left;" src="https://av-eks-blogoptimized.s3.amazonaws.com/boosting10-300x205.png" width="30%"> 

# Thus, the boosting algorithm **combines a number of weak learners to form a strong learner**. The individual models would not perform well on the entire dataset, but they work well for some part of the dataset. Thus, each model actually boosts the performance of the ensemble.

# <a name="ab"></a>
# **AdaBoost** 

# Adaboost, short for "Adaptive Boosting," is a machine learning algorithm used for classification and regression tasks. The main idea behind Adaboost is to **combine several weak learners (classifiers or regression models with accuracy just slightly better than random guessing) to form a strong learner that can make more accurate predictions.**
# 
# The algorithm works by iteratively training a weak learner on the data, and then assigning weights to the data points based on whether the learner got them right or wrong. In each iteration, the weights are updated to give more emphasis on the misclassified data points, so that the next weak learner will focus more on those points.
# 
# The final strong learner is formed by combining the weak learners according to their accuracy, with more weight given to those that perform better. The resulting ensemble model can achieve better accuracy than any of the individual weak learners used in the process.
# 
# 
# **Below are the steps for performing the AdaBoost algorithm:**
# 
# 1. Initially, all observations in the dataset are given equal weights.
# 2. A model is built on a subset of data.
# 3. Using this model, predictions are made on the whole dataset.
# 4. Errors are calculated by comparing the predictions and actual values.
# 5. While creating the next model, higher weights are given to the data points which were predicted incorrectly.
# 6. Weights can be determined using the error value. For instance, higher the error more is the weight assigned to the observation.
# 7. This process is repeated until the error function does not change, or the maximum limit of the number of estimators is reached.

# **Sample Code**

# In[28]:


# SPLITTING THE DATASET
x = df.drop('target', axis = 1)
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[29]:


from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(random_state=1)
model.fit(x_train, y_train)
model.score(x_test,y_test)


# Sample code for regression problem:

# In[30]:


from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor()
model.fit(x_train, y_train)
model.score(x_test,y_test)


# **Parameters**
# 
# - base_estimators: It helps to specify the type of base estimator, that is, the machine learning algorithm to be used as base learner.
# - n_estimators: 
#     - It defines the number of base estimators.
#     - The default value is 10, but you should keep a higher value to get better performance.
# - learning_rate:
#     - This parameter controls the contribution of the estimators in the final combination.
#     - There is a trade-off between learning_rate and n_estimators.
# - max_depth:
#     - Defines the maximum depth of the individual estimator.
#     - Tune this parameter for best performance.
# - n_jobs:
#     - Specifies the number of processors it is allowed to use.
#     - Set value to -1 for maximum processors allowed.
# - random_state:
#     - An integer value to specify the random data split.
#     - A definite value of random_state will always produce same results if given with same parameters and training data.

# <a name="gbm"></a>
# **Gradient Boosting (GBM)**

# GBM (Gradient Boosting) is a machine learning algorithm used for supervised learning tasks, such as regression and classification. 
# 
# It is an ensemble method that combines multiple weak learners (usually decision trees) to form a strong learner that can make more accurate predictions. The algorithm works by iteratively adding decision trees to the model, with each tree attempting to correct the errors made by the previous tree. GBM uses a loss function to evaluate the performance of the model and applies gradient descent optimization to minimize the loss function. The learning rate parameter controls how fast the model updates its predictions based on errors, and it can be adjusted to balance between accuracy and training speed.
# 
# GBM is also known as Gradient Boosting Machines and is widely used in industrial applications and machine learning competitions. The name "Gradient Boosting" comes from the algorithm's use of a gradient descent procedure to minimize the loss when adding new learners to the ensemble.

# <img style="float: left;" src="https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/xgboost_illustration.png" width="70%"> 

# **Below are the steps for performing the AdaBoost algorithm:**
# 
# 1. Initialize the target variable and the predictors, and divide the dataset into training and validation sets.
# 2. Set the number of estimators (decision trees) and the learning rate (the amount each tree contributes to the final prediction).
# 3. Assign equal weights to all observations in the training set.
# 4. Build a decision tree model on a subset of the training data. The model will predict the target variable based on the predictor variables.
# 5. Use this model to make predictions on the entire training set, and calculate the errors by comparing the predictions to the actual values.
# 6. Increase the weights of the observations that were predicted incorrectly, using a weight function that can be defined based on the error value.
# 7. Build the next decision tree model on the updated training set, using the new weights for each observation.
# 8. Repeat steps 5-7 for the desired number of estimators, updating the weights at each step and using the previous models to correct the errors.
# 9. Combine the predictions of all the decision trees to create the final model, using the learning rate to control the contribution of each tree.
# 10. Evaluate the performance of the model on the validation set, and tune the hyperparameters as needed to optimize the accuracy.
# 11. Once the desired level of accuracy is achieved or the maximum number of estimators is reached, the training process is complete, and the model can be used for predictions on new data.

# **Sample Code :**

# In[31]:


from sklearn.ensemble import GradientBoostingClassifier
model= GradientBoostingClassifier(learning_rate=0.01,random_state=1)
model.fit(x_train, y_train)
model.score(x_test,y_test)


# Sample code for regression problem:

# In[32]:


from sklearn.ensemble import GradientBoostingRegressor
model= GradientBoostingRegressor()
model.fit(x_train, y_train)
model.score(x_test,y_test)


# **Parameters**

# - min_samples_split:
#     - Defines the minimum number of samples (or observations) which are required in a node to be considered for splitting.
#     - Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
# - min_samples_leaf:
#     - Defines the minimum samples required in a terminal or leaf node.
#     - Generally, lower values should be chosen for imbalanced class problems because the regions in which the minority class will be in the majority will be very small.
# - min_weight_fraction_leaf:
#     - Similar to min_samples_leaf but defined as a fraction of the total number of observations instead of an integer.
# - max_depth:
#     - The maximum depth of a tree.
#     - Used to control over-fitting as higher depth will allow the model to learn relations very specific to a particular sample.
#     - Should be tuned using CV.
# - max_leaf_nodes:
#     - The maximum number of terminal nodes or leaves in a tree.
#     - Can be defined in place of max_depth. Since binary trees are created, a depth of ‘n’ would produce a maximum of 2^n leaves.
#     - If this is defined, GBM will ignore max_depth.
# - max_features: 
#     - The number of features to consider while searching for the best split. These will be randomly selected.
#     - As a thumb-rule, the square root of the total number of features works great but we should check up to 30-40% of the total number of features.
#     - Higher values can lead to over-fitting but it generally depends on a case to case scenario.

# <a name="xgb"></a>
# **XGBoost**

# XGBoost (extreme Gradient Boosting) is an advanced implementation of the gradient boosting algorithm. XGBoost has proved to be a highly effective ML algorithm, extensively used in machine learning competitions and hackathons. XGBoost has high predictive power and is almost 10 times faster than the other gradient boosting techniques. It also includes a variety of regularization which reduces overfitting and improves overall performance. Hence it is also known as ‘regularized boosting‘ technique.
# 
# 
# **How XGBoost is comparatively better than other techniques**:
# - Regularization: XGBoost integrates L1 and L2 regularization in its objective function to avoid overfitting, unlike the standard GBM implementation.
# - Parallel processing: XGBoost has a built-in parallel processing capability that allows it to run faster than GBM. It also supports implementation on Hadoop.
# - High flexibility: XGBoost enables users to define custom optimization objectives and evaluation criteria, which adds a new dimension to the model.
# - Handling missing values: XGBoost has an in-built routine to handle missing data, making it easy to deal with incomplete datasets.
# - Tree pruning: XGBoost makes splits up to the max_depth specified and then starts pruning the tree backwards, removing splits beyond which there is no positive gain.
# - Built-in cross-validation: XGBoost allows users to run cross-validation at each iteration of the boosting process, enabling them to get the optimum number of boosting iterations in a single run.
# - System optimization: XGBoost employs various algorithm enhancements, such as efficient handling of missing data and parallelized tree building, to speed up the training process significantly.

# Since XGBoost takes care of the missing values itself, you do not have to impute the missing values. You can skip the step for missing value imputation from the code mentioned above. 

# In[33]:


import xgboost as xgb
model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
model.fit(x_train, y_train)
model.score(x_test,y_test)


# Sample code for regression problem:

# In[34]:


import xgboost as xgb
model=xgb.XGBRegressor()
model.fit(x_train, y_train)
model.score(x_test,y_test)


# **Parameters:**
# - nthread:
#     - This is used for parallel processing and the number of cores in the system should be entered..
#     If you wish to run on all cores, do not input this value. The algorithm will detect it automatically.
# - eta:
#     - Analogous to learning rate in GBM.
#     - Makes the model more robust by shrinking the weights on each step.
# - min_child_weight:
#     - Defines the minimum sum of weights of all observations required in a child.
#     - Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
# - max_depth:
#     - It is used to define the maximum depth.
#     - Higher depth will allow the model to learn relations very specific to a particular sample.
# - max_leaf_nodes:
#     - The maximum number of terminal nodes or leaves in a tree.
#     - Can be defined in place of max_depth. Since binary trees are created, a depth of ‘n’ would produce a maximum of 2^n leaves.
#     - If this is defined, GBM will ignore max_depth.
# - gamma: 
#     - A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split.
#     - Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.
# - subsample: 
#     - Same as the subsample of GBM. Denotes the fraction of observations to be randomly sampled for each tree.
#     - Lower values make the algorithm more conservative and prevent overfitting but values that are too small might lead to under-fitting.
# - colsample_bytree:
#     - It is similar to max_features in GBM.
#     - Denotes the fraction of columns to be randomly sampled for each tree.

# <a name="lgbm"></a>
# **Light GBM**

# Light GBM beats all the other algorithms **when the dataset is extremely large**. Compared to the other algorithms, **Light GBM takes lesser time to run on a huge dataset**.
# 
# LightGBM is a gradient boosting framework that uses tree-based algorithms and follows leaf-wise approach while other algorithms work in a level-wise approach pattern. The images below will help you understand the difference in a better way.
# 
# Leaf-wise growth may cause over-fitting on smaller datasets but that can be avoided by using the ‘max_depth’ parameter for learning. 

# <img style="float: left;" src="https://cdn.analyticsvidhya.com/wp-content/uploads/2017/06/11194110/leaf.png" width="35%"> 

# <img style="float: left;" src="https://cdn.analyticsvidhya.com/wp-content/uploads/2017/06/11194227/depth.png" width="45%"> 

# **Sample Code**

# In[35]:


import lightgbm as lgb
train_data=lgb.Dataset(x_train,label=y_train)
#define parameters
params = {'learning_rate':0.001}
model= lgb.train(params, train_data, 100) 
y_pred=model.predict(x_test)
for i in range(0,185):
    if y_pred[i]>=0.5: 
        y_pred[i]=1
else: 
    y_pred[i]=0


# Sample code for regression problem:

# In[36]:


import lightgbm as lgb
train_data=lgb.Dataset(x_train,label=y_train)
params = {'learning_rate':0.001}
model= lgb.train(params, train_data, 100)
from sklearn.metrics import mean_squared_error
rmse=mean_squared_error(y_pred,y_test)**0.5


# **Parameters**
# 
# - num_iterations:
#     - It defines the number of boosting iterations to be performed.
# - num_leaves :
#     - This parameter is used to set the number of leaves to be formed in a tree.
#     - In case of Light GBM, since splitting takes place leaf-wise rather than depth-wise, num_leaves must be smaller than 2^(max_depth), otherwise, it may lead to overfitting.
# - min_data_in_leaf :
#     - A very small value may cause overfitting.
#     - It is also one of the most important parameters in dealing with overfitting.
# - max_depth:
#     - It specifies the maximum depth or level up to which a tree can grow.
#     - A very high value for this parameter can cause overfitting.
# - bagging_fraction:
#     - It is used to specify the fraction of data to be used for each iteration.
#     - This parameter is generally used to speed up the training.
# - max_bin :
#     - Defines the max number of bins that feature values will be bucketed in.
#     - A smaller value of max_bin can save a lot of time as it buckets the feature values in discrete bins which is computationally inexpensive.

# <a name="cb"></a>
# **CatBoost**

# Handling categorical variables is a tedious process, especially when you have a large number of such variables. *When your categorical variables have too many labels (i.e. they are highly cardinal), performing one-hot-encoding on them exponentially increases the dimensionality and it becomes really difficult to work with the dataset*.
# 
# **CatBoost can automatically deal with categorical variables and does not require extensive data preprocessing like other machine learning algorithms**.
# 
# CatBoost algorithm effectively deals with categorical variables. Thus, you should not perform one-hot encoding for categorical variables. Just load the files, impute missing values, and you’re good to go.

# **Sample Code**

# In[37]:


from catboost import CatBoostClassifier
model=CatBoostClassifier()
categorical_features_indices = np.where(df.dtypes != np.float)[0]
model.fit(x_train,y_train,cat_features=([ 0,  1, 2, 3, 4, 10]),eval_set=(x_test, y_test))
model.score(x_test,y_test)


# Sample code for regression problem:

# In[38]:


from catboost import CatBoostRegressor
model=CatBoostRegressor()
categorical_features_indices = np.where(df.dtypes != np.float)[0]
model.fit(x_train,y_train,cat_features=([ 0,  1, 2, 3, 4, 10]),eval_set=(x_test, y_test))
model.score(x_test,y_test)


# **Another Example**

# In[39]:


# importing required libraries
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# read the train and test dataset
train_data = pd.read_csv('/Users/crystal/Desktop/Random Forest/df_03_train.csv')
test_data = pd.read_csv('/Users/crystal/Desktop/Random Forest/df_03_test.csv')

# shape of the dataset
print('Shape of training data :',train_data.shape)
print('Shape of testing data :',test_data.shape)


# In[40]:


# Now, we have used a dataset which has more categorical variables
# hr-employee attrition data where target variable is Attrition 

# seperate the independent and target variable on training data
train_x = train_data.drop(columns=['Attrition'],axis=1)
train_y = train_data['Attrition']

# seperate the independent and target variable on testing data
test_x = test_data.drop(columns=['Attrition'],axis=1)
test_y = test_data['Attrition']


# In[41]:


# find out the indices of categorical variables
categorical_var = np.where(train_x.dtypes != np.float)[0]
print('\nCategorical Variables indices : ',categorical_var)


# In[42]:


model = CatBoostClassifier(iterations=50)

# fit the model with the training data
model.fit(train_x,train_y,cat_features = categorical_var,plot=False)
print('\n Model Trainied')

# predict the target on the train dataset
predict_train = model.predict(train_x)
print('\nTarget on train data',predict_train) 

# Accuray Score on train dataset
accuracy_train = accuracy_score(train_y,predict_train)
print('\naccuracy_score on train dataset : ', accuracy_train)


# In[43]:


# predict the target on the test dataset
predict_test = model.predict(test_x)
print('\nTarget on test data',predict_test) 

# Accuracy Score on test dataset
accuracy_test = accuracy_score(test_y,predict_test)
print('\naccuracy_score on test dataset : ', accuracy_test)


# **Parameters**
# 
# - loss_function: Defines the metric to be used for training.
# - iterations:
#     - The maximum number of trees that can be built.
#     - The final number of trees may be less than or equal to this number.
# - learning_rate:
#     - Defines the learning rate.
#     - Used for reducing the gradient step.
# - border_count:
#     - It specifies the number of splits for numerical features.
#     - It is similar to the max_bin parameter.
# - depth: Defines the depth of the trees.
# - random_seed:
#     - This parameter is similar to the ‘random_state’ parameter we have seen previously.
#     - It is an integer value to define the random seed for training.

# <a name="s"></a>
# #### 3. Stacking

# **Definition :** Stacking is a popular ensemble learning technique that involves combining multiple individual models to improve overall prediction performance. 
# 
# The basic idea behind stacking is to **train several base models on the same dataset, then use their predictions as inputs to a meta-model**. The meta-model can be trained on the same dataset or a different dataset, using the base model predictions as input features. Once the meta-model is trained, it can be used to predict the final outcome on new data.
# 
# **Below is a step-wise explanation for a simple stacked ensemble**

# 1. The train set is split into 10 parts.

# <img style="float: left;" src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/05/image-11-768x555.png" width="35%"> 

# 2. A base model (suppose a decision tree) is fitted on 9 parts and predictions are made for the 10th part. This is done for each part of the train set.

# <img style="float: left;" src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/05/image-10-768x638.png" width="30%"> 

# 3. The base model (in this case, decision tree) is then fitted on the whole train dataset.

# 4. Using this model, predictions are made on the test set.

# <img style="float: left;" src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/05/image-2-768x577.png" width="35%"> 

# 5. Steps 2 to 4 are repeated for another base model (say knn) resulting in another set of predictions for the train set and test set.

# <img style="float: left;" src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/05/image-3-768x573.png" width="35%"> 

# 6. The predictions from the train set are used as features to build a new model.

# <img style="float: left;" src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/05/image12.png" width="25%"> 

# 7. This model is used to make final predictions on the test prediction set.

# **Sample Code**

# In[44]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd


# In[45]:


# SPLITTING THE DATASET
df = pd.read_csv("/Users/crystal/Desktop/Random Forest/heart.csv")
x = df.drop('target', axis = 1)
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[46]:


# Split train and test data into two parts
train1, train2, y_train1, y_train2 = train_test_split(x_train, y_train, test_size=0.5, random_state=1)

# Train and predict on the first base model
model1 = DecisionTreeClassifier(random_state=1)
model1.fit(train1, y_train1)
train_pred1 = model1.predict_proba(train2)[:, 1]
test_pred1 = model1.predict_proba(x_test)[:, 1]

# Train and predict on the second base model
model2 = KNeighborsClassifier()
model2.fit(train1, y_train1)
train_pred2 = model2.predict_proba(train2)[:, 1]
test_pred2 = model2.predict_proba(x_test)[:, 1]

# Only select rows that correspond to the same data points in train_pred1 and test_pred1
train_pred2 = train_pred2[train2.index.isin(train1.index)]
test_pred2 = test_pred2[x_test.index.isin(train1.index)]

# Combine predictions from base models into a single dataframe
train_pred1 = pd.DataFrame(train_pred1)
train_pred2 = pd.DataFrame(train_pred2)
test_pred1 = pd.DataFrame(test_pred1)
test_pred2 = pd.DataFrame(test_pred2)

df = pd.concat([train_pred1, train_pred2], axis=1)
df_test = pd.concat([test_pred1, test_pred2], axis=1)

# Fill any missing values with 0
df.fillna(0, inplace=True)
df_test.fillna(0, inplace=True)

# Train a logistic regression model on the stacked predictions
model = LogisticRegression()
model.fit(df, y_train2)

# Make predictions on the test set using the stacked model
y_pred = model.predict(df_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score of the stacked model:", accuracy)


# <a name="bl"></a>
# #### 4. Blending

# **Definition :** Blending follows the same approach as stacking but **uses only a holdout (validation) set from the train set to make predictions**. In other words, unlike stacking, the predictions are made on the holdout set only. The holdout set and the predictions are used to build a model which is run on the test set. Here is a detailed explanation of the blending process:

# **Below is a step-wise explanation for a simple stacked ensemble**

# 1. The train set is split into training and validation sets.

# <img style="float: left;" src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/05/image-7-768x579.png" width="25%">

# 2. Model(s) are fitted on the training set.

# 3. The predictions are made on the validation set and the test set.

# <img style="float: left;" src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/05/image-5-768x582.png" width="25%">

# 4. The validation set and its predictions are used as features to build a new model.

# 5. This model is used to make final predictions on the test and meta-features.

# **Sample Code**

# We’ll build two models, decision tree and knn, on the train set in order to make predictions on the validation set.

# In[47]:


# SPLITTING THE DATASET
df = pd.read_csv("/Users/crystal/Desktop/Random Forest/heart.csv")
x = df.drop('target', axis=1)
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[48]:


# Split the training data into three parts: train1, train2, and val
train1, train2, y_train1, y_train2 = train_test_split(x_train, y_train, test_size=0.5, random_state=1)
train2, val, y_train2, y_val = train_test_split(train2, y_train2, test_size=0.5, random_state=1)

# Train and predict on the first base model
model1 = DecisionTreeClassifier(random_state=1)
model1.fit(train1, y_train1)
train_pred1 = model1.predict_proba(train2)[:, 1]
val_pred1 = model1.predict_proba(val)[:, 1]
test_pred1 = model1.predict_proba(x_test)[:, 1]

# Train and predict on the second base model
model2 = KNeighborsClassifier()
model2.fit(train1, y_train1)
train_pred2 = model2.predict_proba(train2)[:, 1]
val_pred2 = model2.predict_proba(val)[:, 1]
test_pred2 = model2.predict_proba(x_test)[:, 1]

# Combine predictions from base models into a single dataframe
train_pred1 = pd.DataFrame(train_pred1)
train_pred2 = pd.DataFrame(train_pred2)
val_pred1 = pd.DataFrame(val_pred1)
val_pred2 = pd.DataFrame(val_pred2)
test_pred1 = pd.DataFrame(test_pred1)
test_pred2 = pd.DataFrame(test_pred2)

df_train = pd.concat([train_pred1, train_pred2], axis=1)
df_val = pd.concat([val_pred1, val_pred2], axis=1)
df_test = pd.concat([test_pred1, test_pred2], axis=1)

# Fill any missing values with 0
df_train.fillna(0, inplace=True)
df_val.fillna(0, inplace=True)
df_test.fillna(0, inplace=True)

# Train a logistic regression model on the stacked predictions
model = LogisticRegression()
model.fit(df_train, y_train2)

# Make predictions on the validation set and test set using the stacked model
val_pred = model.predict(df_val)
test_pred = model.predict(df_test)

# Calculate accuracy
val_accuracy = accuracy_score(y_val, val_pred)
test_accuracy = accuracy_score(y_test, test_pred)
print("Accuracy score on the validation set:", val_accuracy)
print("Accuracy score on the test set:", test_accuracy)

