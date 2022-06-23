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
7) MODELLING and PREDICTING
"""



import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
kfold = StratifiedKFold(n_splits=10)



plt.close('all')
%reset -f

##################  1 CLASSIFYING ################


#reading train and test dataframe
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


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


##################  2 CORRELATING ################


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
plt.close('all')
#so now I have understood several things on my data:
# more females survived than man
#those in first class had a better chance to survive compare with those in the thrid class
# only male that embarked from cherbourg had a better chance to survive than others. Females embarking from southampton and queenstown had a better chance.
#finally people that paid higher fares had a better chance to survive
#people with fewer siblings or a spose hada better chance to survive compare to someone with a lot of children or parents

#now it is time to clean up data. 
#since there are many NA in the cabin and each ticket is quite different from others and therefore not giving any interesting information I have to drop them
train_df= train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df= test_df.drop(['Ticket', 'Cabin'], axis=1)
df = [train_df, test_df]

##################  3 CONVERTING ################

#one important thing to notice is that names contain a title (mr, mrs, etc..) this might be helpful

title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}


for i in range(0,2):
    df[i]['Title']=df[i].Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    #convert the just found title in new categories
    df[i]['Title']=df[i]['Title'].replace(['Lady', 'Countess', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df[i]['Title']=df[i]['Title'].replace('Mlle','Miss')
    df[i]['Title']=df[i]['Title'].replace('Ms','Miss')
    df[i]['Title']=df[i]['Title'].replace('Mme','Mrs')
    #convert title into numbers
    df[i]['Title']=df[i]['Title'].map(title_mapping)
    #with the new feature I can remove the name
df[0]=df[0].drop(['Name', 'PassengerId'],axis=1)
df[1]=df[1].drop(['Name'],axis=1)


#now I should convert categorical features in numbers
sex_mapping = {'female':1, 'male':0}
for i in range(0,2):
    df[i]['Sex']=df[i]['Sex'].map(sex_mapping)


##################  4 REPLACING ################

#it looks like that a good way to replace missing data is to find correlation with other variables and try to replace them according to this
#in this specific case there is a correlation between age, gender and Pclass so we can exploit this to find the missing ages.

# g = sns.FacetGrid(train_df, row='Pclass', col='Sex')
# g.map(plt.hist, 'Age', bins=20)
# g.add_legend()


guess_ages = np.zeros((2,3))
for i in range(0,2):
    for j in range(0,2):
        for k in range(0,3):
            guess_df=df[i][(df[i]['Sex']==j) & (df[i]['Pclass']==k+1)]['Age'].dropna()
            age_guess=guess_df.median()
            guess_ages[j,k] =int(age_guess/0.5 +0.5) *0.5
    for j in range(0,2):
        for k in range(0,3):      
            df[i].loc[df[i].Age.isnull() & (df[i].Sex == j) & (df[i].Pclass == k+1), 'Age'] = guess_ages[j,k]


#now instead of using the actual ages, I will include them into tiers

for i in range(0,2):
    df[i].loc[df[i]['Age'] <= 16, 'Age'] = 0
    df[i].loc[(df[i]['Age'] > 16) & (df[i]['Age'] <= 32), 'Age'] = 1
    df[i].loc[(df[i]['Age'] > 32) & (df[i]['Age'] <= 48), 'Age'] = 2
    df[i].loc[(df[i]['Age'] > 48) & (df[i]['Age'] <= 64), 'Age'] = 3 
    df[i].loc[df[i]['Age'] > 64, 'Age'] = 4                                      
                                
# then I want to combine Prch and Sibsp into a family column
for i in range(0,2):
    df[i]['FamilySize'] = df[i]['SibSp'] + df[i]['Parch'] + 1
    df[i]['Alone'] = 0
    df[i].loc[df[i]['FamilySize']==1, 'Alone']=1

   

train_df=df[0].drop(['SibSp', 'Parch', 'FamilySize'],axis=1)
test_df=df[1].drop(['SibSp', 'Parch', 'FamilySize'],axis=1)
df = [train_df, test_df]

emb_mapping = {'S':0, 'C':1, 'Q':2}

#embarked has only two missing data so I take the most common one and I substitute it to those missing
freq=train_df.Embarked.dropna().mode()[0]
for i in range(0,2):
    df[i]['Embarked'] = df[i]['Embarked'].fillna(freq)
#then I convert S C and Q into numbers
    df[i]['Embarked']=df[i]['Embarked'].map(emb_mapping).astype(int)
    
#same analysis for the fare
df[1]['Fare'].fillna(df[1]['Fare'].dropna().median(), inplace=True)
df[1]['FareTier'] = pd.qcut(df[1]['Fare'],4)

for i in range(0,2):
    df[i].loc[df[i]['Fare']<= 7.91, 'Fare'] = 0
    df[i].loc[(df[i]['Fare']> 7.91) & (df[i]['Fare']<=14.454), 'Fare'] = 1
    df[i].loc[(df[i]['Fare']> 14.454) & (df[i]['Fare']<=31), 'Fare'] = 2
    df[i].loc[df[i]['Fare']> 31, 'Fare'] = 3
df[1]=df[1].drop(['FareTier'],axis=1)   

train_df=df[0]
test_df=df[1]

#to show correlation matrix
#sns.heatmap(train_df.corr(), annot=True)

##################  7 MODELLING ################

x_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']
x_test = test_df.drop('PassengerId', axis=1)

#try to normalize value
scaler=StandardScaler()
scaler.fit(train_df)
new = pd.DataFrame(scaler.transform(train_df), columns=train_df.columns)
x_train = new.drop('Survived', axis=1)


scaler.fit(test_df)
new = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)
x_test = new.drop('PassengerId', axis=1)

### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}
gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)
gsSVMC.fit(x_train,y_train)
SVMC_best = gsSVMC.best_estimator_
print('Support Vector Machine: '+ str(round(SVMC_best .score(x_train, y_train)*100,2))+ ' %')

#extra tree classifier
ExtC = ExtraTreesClassifier()
## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}
gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)
gsExtC.fit(x_train,y_train)
ExtC_best = gsExtC.best_estimator_
print('Extra trees classifier: '+ str(round(ExtC_best.score(x_train, y_train)*100,2))+ ' %')


# Regression tree with AdaBoost
tree = DecisionTreeClassifier()
adatree = AdaBoostClassifier(tree, random_state=7)
ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}
gsadaDTC = GridSearchCV(adatree,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)
gsadaDTC.fit(x_train,y_train)
adatree_best = gsadaDTC.best_estimator_
print('Ada Boost Decision tree: '+ str(round(adatree_best.score(x_train, y_train)*100,2))+ ' %')


# RFC Parameters tunning 
RFC = RandomForestClassifier()
## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}
gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)
gsRFC.fit(x_train,y_train)
RFC_best = gsRFC.best_estimator_
print('Random Forest: '+ str(round(RFC_best.score(x_train, y_train)*100,2))+ ' %')

# Gradient boosting
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }
gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)
gsGBC.fit(x_train,y_train)
GBC_best = gsGBC.best_estimator_
print('Gradient boosting: '+ str(round(GBC_best.score(x_train, y_train)*100,2))+ ' %')

#combined the best ones all together

voting_combined = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
('svc', SVMC_best), ('adac',adatree_best),('gbc',GBC_best)], voting='soft', n_jobs=-1)

voting_combined.fit(x_train, y_train)
y_pred=voting_combined.predict(x_test)
print('Combined: '+ str(round(voting_combined.score(x_train, y_train)*100,2))+ ' %')

submission =pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": y_pred})
submission.to_csv('submission_andrea.csv', index=False)




# # #logistic regression
# logreg=LogisticRegression()
# logreg.fit(x_train, y_train)
# y_pred=logreg.predict(x_test)
# print('Logistic regression: '+ str(round(logreg.score(x_train, y_train)*100,2))+ ' %')

# # KNN
# knn=KNeighborsClassifier(n_neighbors=2)
# knn.fit(x_train, y_train)
# y_pred=knn.predict(x_test)
# print('KNN: '+ str(round(knn.score(x_train, y_train)*100,2))+ ' %')

# # Naive Bayes
# gaussian=GaussianNB()
# gaussian.fit(x_train, y_train)
# y_pred=gaussian.predict(x_test)
# print('Naive Bayes: '+ str(round(gaussian.score(x_train, y_train)*100,2))+ ' %')

# #Perceptron
# perceptron = Perceptron()
# perceptron.fit(x_train, y_train)
# y_pred=perceptron.predict(x_test)
# print('Perceptron: '+ str(round(perceptron.score(x_train, y_train)*100,2))+ ' %')


# # Stochastic Gradient Descent
# sgd = SGDClassifier()
# sgd.fit(x_train, y_train)
# y_pred = sgd.predict(x_test)
# print('Stochastic Gradient Descent: '+ str(round(sgd.score(x_train, y_train)*100,2))+ ' %')















# #trying to make prediction with a neural network (two hidden layer, 12 and 8 with relu activation function and sigmoid for output)

# model = Sequential()
# model.add(Dense(12, activation='relu', kernel_initializer='random_normal'))
# kernel_initializer='random_normal'
# model.add(Dense(8, activation='relu', kernel_initializer='random_normal'))
# kernel_initializer='random_normal'
# model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # saving only the best one

# save_best = ModelCheckpoint('best_model.hdf5', save_best_only=True, monitor='val_loss', mode='min')
# #trainint of the NNET, 33 % of validation, 200 epochs, batch size of 10
# model_trained = model.fit(x_train, y_train, validation_split=0.33, epochs=200, batch_size=10, callbacks=save_best)
# plt.close('all')
# plt.figure(1)
# plt.subplot(1,2,1)
# plt.plot(model_trained.history['accuracy'], label='Train', color='r')
# plt.plot(model_trained.history['val_accuracy'], label='Validation',color='b')
# plt.title('model_accuracy')
# plt.xlabel('epoch')
# plt.ylabel('accuracy (%)')
# plt.legend()
# plt.subplot(1,2,2)
# plt.plot(model_trained.history['loss'],label='Train', color='r')
# plt.plot(model_trained.history['val_loss'], label='Validation',color='b')
# plt.title('model_loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend()


# _, accuracy = model.evaluate(x_train, y_train)
# print('neural network: '+ str(round(accuracy*100,2))+ ' %')

# y_pred = model.predict(x_test).reshape(-1)
# predictions = np.zeros(y_pred.shape)
# for i in range(y_pred.shape[0]):
#      if y_pred[i,]>0.5:
#         predictions[i,]=1
#      else:
#         predictions[i,]=0

# submission =pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": predictions.astype(int)})
# submission.to_csv('submission_andrea.csv', index=False)


# #trying to make prediction with a xgboost


# xg = xgb.XGBRegressor(objective ='binary:logistic', learning_rate = 0.1,
#                 max_depth = 5, alpha = 10, n_estimators = 100)

# xg.fit(x_train,y_train)
# y_pred = xg.predict(x_test)
# print('XGBoost: '+ str(round(xg.score(x_train, y_train)*100,2))+ ' %')
# y_pred =(y_pred>0.5)
# print(confusion_matrix(y_test, y_pred))

