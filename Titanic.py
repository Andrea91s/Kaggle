#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 19:12:30 2022

@author: andrea
"""
"""
#this is the first attempt to tackle titanic survivors predictions
#seven steps should be followed
1) CLASSIFYING: classify and categorize the samples, therefore understanding the implications of correlation of different classes
2) CORRELATING: understand which features contribute the most to the solution, both numerical and categorical ones
3) CONVERTING: prepare data for the model that it has to be used (e.g. converting text values in numerical ones)
4) COMPLETING: dealing with missing data
5) CORRECTING:outliers detection and unbalanced data which might require an oversampling
6) CHARTING: plotting
"""

import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

plt.close('all')


#reading train and test dataframe
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
df = [train_df, test_df]

#analyzing data
#1) which are the feature of the dataset?
print(train_df.columns.values)
#2) which are categorical and which numerical?
# Categorical: Survived,Sex,Embarked and Pclass
#Numerical: Age, Fare, Sibsp and Parch
print(train_df.dtypes)
#ticket is king of mixed of numbers and letters. Name is tricky as well since it has many entries with brackets, aka etc...

#this return number of NULL value for each column
print(test_df.isnull().sum())

#this tells us some info on the dataframes
print(train_df.info())
print(test_df.info())

#showing number of survived as a function of class
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#showing number of survived as a function of sex
print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#showing number of survived as a function of sibsp(number of siblings or spouse on board)
print(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#showing number of survived as a function of parch(number of parents or children on board)
print(train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#select only numerical values and plotting theirs distribution

# train_df_num=train_df.select_dtypes({'number'})
# n_cols=4
# n_rows=2
# fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
# for i, column in enumerate(train_df_num):
#     sns.distplot(train_df[column], ax=axes[i//n_cols, i%n_cols])

#distribution of ages as a function of survived and sex

g= sns.FacetGrid(train_df, col='Survived', row='Sex')
g.map(plt.hist, 'Age', bins=30)

#distribution of ages as a function of survived and sex
g= sns.FacetGrid(train_df, col='Survived', row='Pclass')
g.map(plt.hist, 'Age', bins=30)

#this is plotting the mean value of survival as a function of PClass for Male and Female according to the PClass to which they belong to
g = sns.FacetGrid(train_df, col='Embarked')
g.map(sns.pointplot, 'Pclass','Survived', 'Sex', palette='deep')
g.add_legend()

#this barplot for Fare cost as a function of sex, survival rate and embarkment
g = sns.FacetGrid(train_df, row='Embarked', col='Survived')
g.map(sns.barplot, 'Sex', 'Fare')
g.add_legend()

#so now I have understood several things on my data:
# more females survived than man
#those in first class had a better chance to survive compare with those in the thrid class
# only male that embarked from cherbourg had a better chance to survive than others. Females embarking from southampton and queenstown had a better chance.
#finally people that paid higher fares had a better chance to survive
#people with fewer siblings or a spose hada better chance to survive compare to someone with a lot of children or parents

#now it is time to clean up data.






