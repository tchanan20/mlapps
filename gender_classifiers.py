#Numpy deals with large arrays and linear algebra
import numpy as np
# Library for data manipulation and analysis
import pandas as pd 
 
# Metrics for Evaluation of model Accuracy and F1-score
from sklearn.metrics  import f1_score,accuracy_score
 
#Importing the models from scikit-learn library
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
 
# For splitting of data into train and test set
from sklearn.model_selection import train_test_split

# Import dataset
train=pd.read_csv("datasets/gwhtrainingset.csv")
Y = train.Sex  
train.drop(['Sex'], axis=1, inplace=True)
X = train

#Split dataset into test adn train
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=2)

#Testing all imported classifiers to find which gives best results
clfs = [DecisionTreeClassifier(), SVC(), KNeighborsClassifier(), 
        GaussianNB(), MultinomialNB(), SGDClassifier(), 
        RandomForestClassifier(), GradientBoostingClassifier(), LGBMClassifier()]
results = {}
for clf in clfs:
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    results[clf.__class__.__name__] = accuracy_score(y_test,pred)    

L = [[k,v] for [k,v] in results.items()]
sorted(L, key=lambda x:x[1], reverse=True)