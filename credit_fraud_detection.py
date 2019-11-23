#!/usr/bin/env python
# coding: utf-8

# ## Ensemble Learning Model to Predict Credit Card Fraud
# 
# ### Dataset from Kaggle contains 284,807 examples with 31 features.

# ##### Import Pandas, Numpy, Matplotlib, and various Scitkit Learn libraries

# In[213]:


get_ipython().run_line_magic('matplotlib', 'inline')

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import pandas as pd
import numpy as np
import seaborn as sns
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import matplotlib.patches as mpatches

# Accuracy Measurements
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics as metrics
from scipy.stats import skew, norm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Clustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD


# ##### Load Dataset from csv file

# In[183]:


CREDIT_FRAUD_PATH = os.path.join("dataset", "creditcardfraud")

if not os.path.isdir(CREDIT_FRAUD_PATH):
    os.makedirs(CREDIT_FRAUD_PATH)

def load_credit_fraud_data(credit_fraud_path=CREDIT_FRAUD_PATH):
    csv_path = os.path.join(credit_fraud_path, "creditcard.csv")
    
    return pd.read_csv(csv_path, float_precision='round_trip')

credit_fraud = load_credit_fraud_data()


# ##### Visualize the Data

# In[184]:


#credit_fraud.hist(bins=50, figsize=(15,15))
fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = credit_fraud['Amount'].values
time_val = credit_fraud['Time'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

plt.show()


# ### Scale 'Amount' and 'Time'as they are the only attributes which haven't been scaled yet
# ##### Standard Scaler assumes your data is normally distributed within each feature and will scale them such that the distribution is now centered around 0. Because our data is not normally distributed, this isn't the best scaler to use.
# ##### Min-max scaler works well when the standard deviation is small, but is sensitive to outliers.
# ##### Robust Scaler uses interquartile range which is robust to outliers
# 
# ##### Before we decide which scaler to use, lets look at the boxplots to see the outlier situation.

# In[185]:


f, ax = plt.subplots(figsize=(20, 4))
sns.boxplot(x="Class", y="Amount", data=credit_fraud, ax=ax)

plt.show()


# ##### As we can see there are many outliers so it would be best to use RobustScaler.

# In[186]:


robust_scaler = RobustScaler()

credit_fraud['scaled_amount'] = robust_scaler.fit_transform(credit_fraud['Amount'].values.reshape(-1, 1))
credit_fraud['scaled_time'] = robust_scaler.fit_transform(credit_fraud['Time'].values.reshape(-1,1))
credit_fraud.drop(['Time', 'Amount'], axis=1, inplace=True)

scaled_amount = credit_fraud['scaled_amount']
scaled_time = credit_fraud['scaled_time']
credit_fraud.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
credit_fraud.insert(0, 'scaled_amount', scaled_amount)
credit_fraud.insert(1, 'scaled_time', scaled_time)


# #### Split the data up using Stratified K-fold Cross Validation

# In[187]:


credit_X = credit_fraud.drop('Class', axis=1)
credit_y = credit_fraud['Class']

strat_k_split = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in strat_k_split.split(credit_X, credit_y):
    original_X_train, original_X_test = credit_X.iloc[train_index], credit_X.iloc[test_index]
    original_y_train, original_y_test = credit_y.iloc[train_index], credit_y.iloc[test_index]


# ##### Check the label distribution 

# In[188]:


train_unique_labels, train_counts_labels = np.unique(original_y_train, return_counts=True)
test_unique_labels, test_counts_labels = np.unique(original_y_test, return_counts=True)

print(train_counts_labels / len(original_y_train))
print(test_counts_labels / len(original_y_test))

colors = ["#0101DF", "#DF0101"]
sns.countplot('Class', data=credit_fraud, palette=colors)
plt.title("Class Distributions")


# ### Random Undersampling
# ##### Removing data to produce a more balanced dataset

# In[189]:


# Shuffle the data
credit_fraud = credit_fraud.sample(frac=1)

# Determine how many instances of fraud vs. not fraud.
num_fraud, num_non_fraud = credit_fraud["Class"].value_counts()
fraud_instances = credit_fraud.loc[credit_fraud["Class"] == 1]
non_fraud = credit_fraud.loc[credit_fraud["Class"] == 0][:num_non_fraud]

# Concatenate the balanced fraud and non fraud instances 
normal_distributed_credit_fraud = pd.concat([fraud_instances, non_fraud])

# Shuffle the concatenated values
credit_bal = normal_distributed_credit_fraud.sample(frac=1, random_state=42)
credit_bal.head()

# Check the label distributions
sns.countplot('Class', data=credit_bal, palette=colors)
plt.title("Class Distributions")


# #### Correlations

# In[190]:


fig, ax1 = plt.subplots(1, 1, figsize=(20, 10))

correlation = credit_bal.corr()
sns.heatmap(data=correlation, cmap='vlag_r', ax=ax1)
ax1.set_title("Subsample Correlation Matrix")

plt.show()


# #### Use boxplots to display quartiles and outliers on highly correlated features

# In[191]:


correlated_features = ['V2', 'V4', 'V11', 'V19']
correlated_features_negative = ['V10', 'V12', 'V14', 'V16']

# Positive Correlations
fig, axes = plt.subplots(ncols=4, figsize=(20, 4))

sns.boxplot(x='Class', y='V2', data=credit_bal, palette=colors, ax=axes[0])
axes[0].set_title("V2 vs Class Positive Correlation")

sns.boxplot(x='Class', y='V4', data=credit_bal, palette=colors, ax=axes[1])
axes[1].set_title("V4 vs Class Positive Correlation")

sns.boxplot(x='Class', y='V11', data=credit_bal, palette=colors, ax=axes[2])
axes[2].set_title("V11 vs Class Positive Correlation")

sns.boxplot(x='Class', y='V19', data=credit_bal, palette=colors, ax=axes[3])
axes[3].set_title("V19 vs Class Positive Correlation")

plt.show()


# Negative Correlations

fig, axes = plt.subplots(ncols=4, figsize=(20, 4))

sns.boxplot(x='Class', y='V10', data=credit_bal, palette=colors, ax=axes[0])
axes[0].set_title("V10 vs Class Positive Correlation")

sns.boxplot(x='Class', y='V12', data=credit_bal, palette=colors, ax=axes[1])
axes[1].set_title("V12 vs Class Positive Correlation")

sns.boxplot(x='Class', y='V14', data=credit_bal, palette=colors, ax=axes[2])
axes[2].set_title("V14 vs Class Positive Correlation")

sns.boxplot(x='Class', y='V16', data=credit_bal, palette=colors, ax=axes[3])
axes[3].set_title("V16 vs Class Positive Correlation")

plt.show()


# #### Visualize Distributions

# In[192]:


fig, axes = plt.subplots(1, 4, figsize=(20, 6))

v14_fraud_dist = credit_bal['V14'].loc[credit_bal['Class'] == 1].values
sns.distplot(v14_fraud_dist,ax=axes[0], fit=norm, color='#FB8861')
axes[0].set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)

v12_fraud_dist = credit_bal['V12'].loc[credit_bal['Class'] == 1].values
sns.distplot(v12_fraud_dist,ax=axes[1], fit=norm, color='#56F9BB')
axes[1].set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)

v10_fraud_dist = credit_bal['V10'].loc[credit_bal['Class'] == 1].values
sns.distplot(v10_fraud_dist,ax=axes[2], fit=norm, color='#C5B3F9')
axes[2].set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)

v16_fraud_dist = credit_bal['V16'].loc[credit_bal['Class'] == 1].values
sns.distplot(v16_fraud_dist,ax=axes[3], fit=norm, color='#1688BF')
axes[3].set_title('V16 Distribution \n (Fraud Transactions)', fontsize=14)

plt.show()


# #### Remove Outliers

# In[194]:


feature_names = ['V10', 'V12', 'V14', 'V16', 'V2', 'V4', 'V11', 'V19']

v2_fraud_dist = credit_bal['V2'].loc[credit_bal['Class'] == 1].values
v4_fraud_dist = credit_bal['V4'].loc[credit_bal['Class'] == 1].values
v11_fraud_dist = credit_bal['V11'].loc[credit_bal['Class'] == 1].values
v19_fraud_dist = credit_bal['V19'].loc[credit_bal['Class'] == 1].values

features_with_outliers = [v10_fraud_dist, v12_fraud_dist, v14_fraud_dist, v16_fraud_dist,
                         v2_fraud_dist, v4_fraud_dist, v11_fraud_dist, v19_fraud_dist]

def remove_outliers(dataset, outliers_to_remove, names):
    for i in range(len(outliers_to_remove)):
        feature = outliers_to_remove[i]
        name = names[i]
        print(name)
        q25, q75 = np.percentile(feature, 25), np.percentile(feature, 75)
        interquartile_rng = q75 - q25
        cutoff = interquartile_rng * 1.5
        feature_lower, feature_upper = q25 - cutoff, q75 + cutoff  
        print("IQR:", interquartile_rng)
        print("Cutoff: ", cutoff, "\n")
        print("Upper: ", feature_upper)
        print("Lower: ", feature_lower, "\n")
        outliers = [x for x in feature if x < feature_lower or x > feature_upper]
        print(name + ' outliers:{}'.format(outliers), "\n")
        print('Feature ' + name + ' Outliers for Fraud Cases: {}'.format(len(outliers)), "\n \n")
        
        dataset = dataset.drop(dataset[(dataset[name] > feature_upper) | (dataset[name] < feature_lower)].index)
        credit_bal = dataset
    return dataset

credit_bal = remove_outliers(credit_bal, features_with_outliers, feature_names)


# #### View boxplots to check out outlier removal changes

# In[195]:


fig, axes = plt.subplots(1, 3, figsize=(20, 6))

sns.boxplot(x="Class", y="V14", data=credit_bal, ax=axes[0], palette=colors)
axes[0].set_title("V14 Feature \n Reduction of Outliers", fontsize=14)

sns.boxplot(x="Class", y="V12", data=credit_bal, ax=axes[1], palette=colors)
axes[1].set_title("V12 Feature \n Reduction of Outliers", fontsize=14)

sns.boxplot(x="Class", y="V10", data=credit_bal, ax=axes[2], palette=colors)
axes[2].set_title("V10 Feature \n Reduction of Outliers", fontsize=14)

plt.show()


# ### Clustering Algorithms
# ##### t-SNE, PCA, and Truncated SVD

# In[201]:


X = credit_bal.drop('Class', axis=1)
y = credit_bal['Class']

X_reduced_tsne = TSNE(random_state=42).fit_transform(X.values)

X_reduced_pca = PCA(random_state=42).fit_transform(X.values)

X_reduced_svd = TruncatedSVD(algorithm='randomized', random_state=42).fit_transform(X.values)


# #### Visualize Clusters

# In[204]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
f.suptitle('Clustering Algorithms', fontsize=14)

blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')

# t-SNE scatter plot
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax1.set_title('t-SNE', fontsize=14)

ax1.grid(True)

ax1.legend(handles=[blue_patch, red_patch])


# PCA scatter plot
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax2.set_title('PCA', fontsize=14)

ax2.grid(True)

ax2.legend(handles=[blue_patch, red_patch])

# TruncatedSVD scatter plot
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax3.set_title('Truncated SVD', fontsize=14)

ax3.grid(True)

ax3.legend(handles=[blue_patch, red_patch])

plt.show()


# ### Classification

# In[212]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier()
}

for type, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    train_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print("Classifier " + classifier.__class__.__name__, "has training score: ", round(train_score.mean(), 2) * 100)


# ### Use GridSearch to find the best parameters
# ##### Create the grid search parameters for each classifier

# In[224]:


logistic_reg_params = [{"penalty": ['l1'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                       'solver': ['liblinear', 'saga']},
                       {"penalty": ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                       'solver': ['newton-cg', 'sag', 'lbfgs']},
                       {"penalty": ['elasticnet'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                       'solver': ['saga'], 'l1_ratio': [0, 0.25, 0.5, 0.75, 1]}
                      ]

log_reg_grid = GridSearchCV(LogisticRegression(), logistic_reg_params)
log_reg_grid.fit(X_train, y_train)

log_reg = log_reg_grid.best_estimator_

knears_params = {"n_neighbors": [1, 4, 7, 10, 13, 16, 19, 22], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 
                  'weights': ['uniform', 'distance']}

knears_grid = GridSearchCV(KNeighborsClassifier(), knears_params)
knears_grid.fit(X_train, y_train)
knears_neighbors = knears_grid.best_estimator_


# ##### Create a train and test set. These sets may contain null values because we haven't pre-processed them yet. 
# ##### Because the dataset has enough training examples, we will use a 75 / 25 train test split.

# In[295]:


# Add an index column to the dataset.
credit_fraud_with_id = credit_fraud.reset_index()

train_set, test_set = train_test_split(credit_fraud_with_id, train_size=0.75, random_state=42, shuffle=True)
credit_train = train_set.copy()
credit_test = test_set.copy()


# ##### Pre-process methods

# In[296]:


def impute_median(data):
    """Imputation transformer for completing missing numerical values."""
    if isinstance(data, pd.DataFrame):
        imputer = SimpleImputer(strategy="median")
        imputer.fit(data)
        X = imputer.transform(data)
        return pd.DataFrame(X, columns=data.columns, index=data.index)
    else:
        # not a DataFrame
        return None
    
def drop_missing_values(data):
    """Drops data if there are less than 17 non-NA values in the row"""
    if isinstance(data, pd.DataFrame):
        # The row must contain at least 17 non-NA values to stay in the dataset.
        data.dropna(how="any", axis=0, thresh=17)
        return data.fillna("", inplace=False)
    else:
        # not a DataFrame
        return None
    
def log_transform(data):
    """Log Transformation for left skewed data"""
    if isinstance(data, pd.Series):
        # Check skew value. If outside of -1 to 1 range, do the log transform
        skew_value = skew(data)
        if (abs(skew_value) > 1):
            minimum = data.min()
            data_copy = data.copy()

            # Shift right to ensure all positive values
            data_copy += (-1 * minimum) + 2

            # Execute log Transform
            transformed_data = data_copy.transform(np.log)

            # Shift back left
            transformed_data += minimum - 2
            return transformed_data
        else:
            # Not skewed enough to worry about
            return data
    else:
        # not a DataFrame
        return data


# ##### Execute Pre-processing

# In[297]:


credit_transformed = credit_fraud_with_id.copy()
features_to_transform = ["V24", "V26"]
for i in range(len(features_to_transform)):
    feature = features_to_transform[i]
    credit_transformed[feature] = log_transform(credit_transformed[feature])
    
credit_dropped = drop_missing_values(credit_transformed)
credit_fraud_imputed = impute_median(credit_fraud_with_id)


# ##### Stratify data

# In[298]:


strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

#credit_dropped["time_attribute"] = pd.cut(credit_dropped["Time"], 
#                                       bins=[0., 35000, 70000, 105000, 140000, 175000., np.inf],
#                                       labels=[1, 2, 3, 4, 5, 6])

### Figure out scikitlearn issue here.
#for train_index, test_index in strat_split.split(credit_dropped, credit_dropped["time_attribute"]):
#    print(train_index)
#    print(test_index)
#    strat_train_set = credit_dropped.loc[train_index]
#    strat_test_set = credit_dropped.loc[test_index]


# ##### Visualize the data to gain insights

# In[299]:


for i in range(28):
    feature = "V" + str(i + 1)
    credit_train.plot(kind="scatter", x=feature, y="Class", alpha="0.1")


# ### Look at data correlations
# ##### Correlation coefficient ranges from -1 to 1. When it is close to one, it means there is a string positive correlation. 
# ##### Example: if there is a strong positive correlation, the Class value tends to go up when the correlated attribute goes up.
# ##### Example: If there is a strong negative correlation, that means the Class value tends to decrease when the correlated attribute decreases. 
# ##### Note that the correlation coefficient only measure linear correlations! It completely misses out on any nonlinear relationships
# ##### We find that there aren't any strong correlations between the Class attribute and any of the features. V17 has the largest negative correlation at -0.324 and feature V11 has the largest positive correlation at 0.15.

# In[300]:


corr_matrix = credit_train.corr()


# ##### Another way to check for correlations between attributes is to use pandas scatter_matrix() function which plots every numerical attribute against every other numerical attribute. Since there are 32 numerical attributes, we would get 32 * 32 = 1024 plots.
# ##### Try out a few classes below

# In[301]:


attributes = ["Class", "V11", "V4", "V2"]
matrix = scatter_matrix(credit_train[attributes], figsize=(12,8))
credit_train.plot(kind="scatter", x="V11", y="Class", alpha=0.1)


# #### Drop label feature ('Class'), index, and time feature. 
# ##### The time feature is being dropped because it doesn't add any value. This was added for a metric to know when the transactions occured after each other (NOT the time of day it occured).

# In[302]:


to_drop = ["Class", "index", "Time"]
credit_train_X = credit_train.drop(to_drop, axis=1)
credit_train_Y = credit_train["Class"].copy()

credit_test_X = credit_test.drop(to_drop, axis=1)
credit_test_Y = credit_test["Class"].copy()


# #### Train a Random Forest with different number of trees to see accuracy results
# ##### ** Note ** for loop is in range 1 for faster testing

# In[303]:


number_of_trees = [50, 250, 500, 750]
classifiers = []

for i in range(1):
    num_estimators = number_of_trees[i]
    rnd_forest_clf = RandomForestClassifier(n_estimators=num_estimators, n_jobs=-1, oob_score=True)
    
    # Train the random forest
    rnd_forest_clf.fit(credit_train_X, credit_train_Y)
    
    # Make predictions
    y_pred = rnd_forest_clf.predict(credit_test_X)
    
    # Generate Scores
    score = accuracy_score(credit_test_Y, y_pred)
    classifiers.append(rnd_forest_clf)
    misclassified = credit_test_Y[y_pred != credit_test_Y]
    num_test_instances = y_pred.shape[0]
    num_misclassified = misclassified.shape[0]


# #### Gather results from training and testing

# In[304]:


recall_score = recall_score(credit_test_Y, y_pred)
precision_score = precision_score(credit_test_Y, y_pred)
confusion_matrix = confusion_matrix(credit_test_Y, y_pred)
accuracy_score = accuracy_score(credit_test_Y, y_pred)
f1_score = f1_score(credit_test_Y, y_pred)
probabilites = rnd_forest_clf.predict_proba(credit_test_X)
predictions = probabilites[:,1]
fpr, tpr, threshold = metrics.roc_curve(credit_test_Y, predictions)
roc_auc = metrics.auc(fpr, tpr)


# #### Analyze and visualize the results
# ##### Plot the ROC Area Under Curve Score, then display the F1 score, a confusion matrix, accuracy score, the OOB score, and finally the model accuracy.
# ##### ** Note ** With classification models, it is important to use testing methods other than accuracy to determine how good your model is predicting.

# In[309]:


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.5, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

print("F1 Score:", f1_score)
print("Confusion Matrix")
print(confusion_matrix)
print("Accuracy Score", accuracy_score)
print("Number of test instances:", num_test_instances)
print("Number of misclassified instances:", num_misclassified)
print("Model Accuracy", ( (num_test_instances - num_misclassified) / num_test_instances))

for y in range(len(classifiers)):
    print("OOB Score: ", classifiers[y].oob_score_)

