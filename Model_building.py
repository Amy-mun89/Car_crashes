#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:21:11 2019

@author: Amy
"""

##### 1. Import data and library
import os
os.getcwd()
os.chdir('/Users/appleuser/Documents/19_2_PM DS lab')

import numpy as np
import pandas as pd
import seaborn as sns 
import pickle
import tarfile

# open a .spydata file
filename = 'dataframes.spydata'
tar = tarfile.open(filename, "r")

# extract all pickled files to the current working directory
tar.extractall()
extracted_files = tar.getnames()
for f in extracted_files:
    if f.endswith('.pickle'):
         with open(f, 'rb') as fdesc:
             data = pickle.loads(fdesc.read())

# or use the spyder function directly:
from spyderlib.utils.iofuncs import load_dictionary
data_dict = load_dictionary(filename)



import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels

pd.set_option('display.expand_frame_repr', False)
testdf = pd.read_pickle("Crashes_Encoded_Amy_002")
testdf.info()

df_selected = testdf.drop(["RD_NO","CRASH_DATE","STREET_NO","INJURIES_TOTAL","INJURIES_FATAL","INJURIES_INCAPACITATING",
                           "INJURIES_NON_INCAPACITATING","INJURIES_REPORTED_NOT_EVIDENT","INJURIES_NO_INDICATION",
                           "INJURIES_UNKNOWN","DAMAGE_$501 - $1,500","DAMAGE_OVER $1,500","MOST_SEVERE_INJURY_INCAPACITATING INJURY",
                           "MOST_SEVERE_INJURY_NO INDICATION OF INJURY","MOST_SEVERE_INJURY_NONINCAPACITATING INJURY",
                           "MOST_SEVERE_INJURY_REPORTED, NOT EVIDENT"], axis=1)
df_selected.info()


##### 2.  Deciding upon the trining set sizes 
### 2-1) define the Models. 

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth = 10)
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(max_depth=5)

# set the train size 

train_sizes = [1,100,500,5000,7175, 9567,11959, 14351, 16743, 19134] # 1,100,500,30,40,50,60,70,80% 
train_sizes2 = [100,500,1000,5000,7175, 9567,11959, 14351, 16743, 19134] # 100,500,1000,5000,30,40,50,60,70,80% 

from sklearn.model_selection import learning_curve

# score the each train sizes 
train_sizes, train_scores, validation_scores = learning_curve(
estimator = clf,
X = df_selected.drop("CRASH_TYPE_NO INJURY / DRIVE AWAY", axis = 1),
y = df_selected["CRASH_TYPE_NO INJURY / DRIVE AWAY"], train_sizes = train_sizes, cv = 10,
scoring = 'accuracy')


### 2-2) Calculate the error rate for classification. 1-accuracy  ====> 20:80

print('Training scores:\n\n', train_scores)
print('\n', '-' * 70) # separator to make the output easy to read
print('\nValidation scores:\n\n', validation_scores)
train_scores_mean = (1-train_scores.mean(axis = 1))*100
validation_scores_mean =(1- validation_scores.mean(axis = 1))*100
print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
print('\n', '-' * 20) # separator
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))


plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Test error')
plt.ylabel('error rate', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Error rates for a Decision tree model', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,80)


### 2-3) Calculate the error rate for regression. MSE 
##  Logistic regression 
train_sizes2 = [100,500,1000,5000,7175, 9567,11959, 14351, 16743, 19134] # 100,500,1000,5000,30,40,50,60,70,80% 
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(copy=True, feature_range=(0, 1))
df_selected_features = df_selected.drop("CRASH_TYPE_NO INJURY / DRIVE AWAY", axis = 1)
scaler.fit(df_selected_features)
df_selected_scaled = scaler.transform(df_selected_features)
train_sizes, train_scores, validation_scores = learning_curve(
estimator = LR,
X = df_selected.drop("CRASH_TYPE_NO INJURY / DRIVE AWAY", axis = 1),
y = df_selected["CRASH_TYPE_NO INJURY / DRIVE AWAY"], train_sizes = train_sizes2, cv = 10,
scoring = 'neg_mean_squared_error')



print('Training scores:\n\n', train_scores)
print('\n', '-' * 70) # separator to make the output easy to read
print('\nValidation scores:\n\n', validation_scores)
train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)
print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
print('\n', '-' * 20) # separator
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))

plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Test error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a Logistic regression model', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,0.5)

##  Random forrest regression
train_sizes = [1,100,500,5000,7175, 9567,11959, 14351, 16743, 19134] # 1,100,500,30,40,50,60,70,80% 

train_sizes, train_scores, validation_scores = learning_curve(
estimator = RFR,
X = df_selected.drop("CRASH_TYPE_NO INJURY / DRIVE AWAY", axis = 1),
y = df_selected["CRASH_TYPE_NO INJURY / DRIVE AWAY"], train_sizes = train_sizes, cv = 10,
scoring = 'neg_mean_squared_error')

print('Training scores:\n\n', train_scores)
print('\n', '-' * 70) # separator to make the output easy to read
print('\nValidation scores:\n\n', validation_scores)
train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)
print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
print('\n', '-' * 20) # separator
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))

plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Test error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a randomforrest regression model', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,0.8)


data2 = df_selected.drop("CRASH_TYPE_NO INJURY / DRIVE AWAY", axis = 1)
X = data2 # all Features
y = df_selected["CRASH_TYPE_NO INJURY / DRIVE AWAY"] # Target variable
class_names = y

#Split dataset into training set and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 80% training and 20% test


##### 3. Feature selection Analysis

## 1) Filter Method - Correlation matrix

cor = df_selected.corr() 
f, ax = plt.subplots(figsize =(80 , 80)) 
sns.heatmap(cor, ax = ax, cmap ="YlGnBu", linewidths = 0.1, annot=True) 
#Correlation with output variable
cor_target = abs(cor["CRASH_TYPE_NO INJURY / DRIVE AWAY"])
cor_target

#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.05]
relevant_features
related_col=relevant_features.index.tolist()
rel = df_selected[related_col]
rel2 = df_selected[related_col]


##2) chi squre (Univarite analysis)
#The null hypothesis of the Chi-Square test is that
#no relationship exists on the categorical variables in the population; they are
#independent. => hight score: higly dependent. 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(rel,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(rel.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features 
selected_col_chi = featureScores.nlargest(10,'Score')["Features"].tolist()


rel = rel.drop("CRASH_TYPE_NO INJURY / DRIVE AWAY", axis=1)

##3) feature importance

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(rel,y)

model.fit(X,y)


X.isnull().sum()
X = X.fillna(X.mode().iloc[0])
y.isnull().sum()

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=rel.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
selected_col_Importance = feat_importances.nlargest(10).index.tolist()

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()
selected_col_Importance = feat_importances.nlargest(20).index.tolist()




''' ##4) boruta
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

XX = X.to_records(index=False)
np.asarray(XX)
XX
yy = y.to_numpy()
yy
# find all relevant features - 5 features should be selected
feat_selector.fit(XX, yy)
# check selected features - first 5 features are selected
feat_selector.support_

# check ranking of features
feat_selector.ranking_

X_filtered = feat_selector.transform(XX) '''


##### 4. Model Building
### 4-1 ) Decision Tree

# Load libraries
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

class_names = y
X= rel[selected_col_Importance]
y= df_selected["CRASH_TYPE_NO INJURY / DRIVE AWAY"]

#Split dataset into training set and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

X.isnull().sum()
X = X.fillna(X.mode().iloc[0])
y.isnull().sum()
y=y.fillna(0)
y = y.fillna(y.mode().iloc[0])


### 4-1.a) find hyper parameter: max depths
# decide the max depths
from sklearn.metrics import roc_curve, auc

max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []

for max_depth in max_depths:
   dt = DecisionTreeClassifier(max_depth=max_depth)
   dt.fit(X_train, y_train)
   train_pred = dt.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   # Add auc score to previous train results
   train_results.append(roc_auc)
   y_pred = dt.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   # Add auc score to previous test results
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, "b", label= "Train AUC")
line2, = plt.plot(max_depths, test_results, "r", label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("Tree depth")
plt.show()





   
### 4-1.b) find hyper parameter: min samples leafs
   
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
train_results = []
test_results = []
for min_samples_leaf in min_samples_leafs:
   dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
   dt.fit(X_train, y_train)
   train_pred = dt.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = dt.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
   
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_leafs, train_results, "b", label="Train AUC")
line2, = plt.plot(min_samples_leafs, test_results, "r", label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("min samples leaf")
plt.show()




### 4-1.c) find hyper parameter: min samples splits

min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
   dt = DecisionTreeClassifier(min_samples_split=min_samples_split)
   dt.fit(X_train, y_train)
   train_pred = dt.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds =    roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = dt.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_splits, train_results, "b", label="Train AUC")
line2, = plt.plot(min_samples_splits, test_results, "r", label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("min samples split")
plt.show()




### 4-1.d) find hyper parameter: max features

max_features = list(range(1,X.shape[1]))
train_results = []
test_results = []
for max_feature in max_features:
   dt = DecisionTreeClassifier(max_features=max_feature)
   dt.fit(X_train, y_train)
   train_pred = dt.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = dt.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_features, train_results, "b", label="Train AUC")
line2, = plt.plot(max_features, test_results, "r", label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("max features")
plt.show()




# Create Decision Tree classifer object

clf = DecisionTreeClassifier(max_depth = 11,min_samples_split = 10,
                             min_samples_leaf =10,max_leaf_nodes=30)

clf = DecisionTreeClassifier(max_depth = 10)


related_col=relevant_features.index.tolist()
selected_col_Importance = feat_importances.nlargest(10).index.tolist()
selected_col_chi = featureScores.nlargest(10,'Score')["Features"].tolist()


X = X[selected_col_Importance]
class_names = y

#Split dataset into training set and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)


#Predict the response for test dataset
y_pred = clf.predict(X_test)



from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix
from sklearn import tree
import graphviz

dot_data = tree.export_graphviz(clf, class_names=['No Injuries', 'Injuries'],
                                out_file=None,filled=True, rounded=True, proportion=True, label="root",
                                feature_names = X.columns)
graph = graphviz.Source(dot_data)

graph # show the decision tree



### 4-2 ) RandomForrestRegression
selected_col_Importance = feat_importances.nlargest().index.tolist()
selected_col_chi = featureScores.nlargest(10,'Score')["Features"].tolist()

X = rel[selected_col_Importance] # Features
X = rel[selected_col_chi] # Features

class_names = y

#Split dataset into training set and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# Train Decision Tree Classifer
RFR = RFR.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = RFR.predict(X_test)
y_pred = y_pred.round()

### 4-3 ) LogisticRegression
selected_col_Importance = feat_importances.nlargest().index.tolist()
selected_col_chi = featureScores.nlargest(10,'Score')["Features"].tolist()


X = data2[selected_col_Importance] # Features
class_names = y

#Split dataset into training set and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# Train Decision Tree Classifer
LR = LR.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = LR.predict(X_test)
y_pred = y_pred.round()




##### 5. Model Evaluation

print("<Model Evaluation>","\n",metrics.classification_report(y_test, y_pred),"\n"
      "<AUC Score>","\n",metrics.roc_auc_score(y_test,y_pred))
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred)
auc=metrics.roc_auc_score(y_test,y_pred)
print(plt.plot([0, 1], [0, 1], linestyle='--'),plt.plot(fpr, tpr, marker='.'),
plt.plot(fpr, tpr, marker='.'),
plt.title('ROC Curve'),
plt.xlabel('TPR'),
plt.ylabel('FPR'),
plt.grid(),
plt.legend(["AUC=%.3f"%auc]))

confusion_matrix(y_test, y_pred)




#Predict the response for test dataset for regression models
y_pred = clf.predict(X_test)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
from sklearn.metrics import mean_squared_error 
mean_squared_error(y_test, y_pred)
metrics.accuracy_score(y_test, y_pred.round(), normalize=False)



# Note that in binary classification, recall of the positive class is also known as “sensitivity”; 
# recall of the negative class is “specificity”.
# see the training error and test error




#Confusion Matrix 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def plot_confusion_matrix(y_test2, y_pred2, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_test2, y_pred2)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_test, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Test label',
           xlabel='Predicted label')


plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, Decision Tree')


