# --------------------------------------------------------
# ML Solution for the Kaggle ML problem 
# Adapted from https://www.kaggle.com/code/preejababu/titanic-data-science-solutions#Acquire-data
# --------------------------------------------------------
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Annotating the path to the testing and training sets 
training_path = 'train.csv'
testing_path = 'test.csv'

training_set = pd.read_csv(training_path)
testing_set = pd.read_csv(testing_path)
combine = [training_set, testing_set]

# ------ Data analysis through description ------
# Shows which features are in your data set
print(training_set.columns.values)
print('\n')

# preview the data
# this gives us the first couple entries...
print(training_set.head())
print('\n')

# this gives us the last couple entries
print(training_set.tail())
print('\n')

# gives us information about our data
training_set.info()
print('_'*40)
testing_set.info()
print('\n')

print(training_set.describe())
print('\n')
# ---------------------------------------------

# ------ Data analysis through pivoting features ------
print(training_set[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('\n')
print(training_set[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Sex', ascending=False))
print('\n')
print(training_set[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='SibSp', ascending=False))
print('\n')
print(training_set[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Parch', ascending=False))
# ----------------------------------------------------

# ------ Analyze by visualizing data ------
# Survival based on age
survival_plot = sns.FacetGrid(training_set, col='Survived')
survival_plot.map(plt.hist, 'Age', bins=20)

# Survival based on Pclass and Age
pclass_survival = sns.FacetGrid(training_set, col='Survived', row='Pclass', size=2.2, aspect=1.6)
pclass_survival.map(plt.hist, 'Age', alpha=.5, bins=20)
pclass_survival.add_legend();

# Survival based on sex and age
sex_survival = sns.FacetGrid(training_set, col='Survived', row='Sex', size=2.2, aspect=1.6)
sex_survival.map(plt.hist, 'Age', alpha=.5, bins=20)
sex_survival.add_legend();

# Survival based on Pclass and port of Embarkal
embarkal_survival = sns.FacetGrid(training_set, row='Embarked', size=2.2, aspect=1.6)
embarkal_survival.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
embarkal_survival.add_legend()

# Survival based off of different attributes
grid = sns.FacetGrid(training_set, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
# -------------------------------------

# ------ Wrangle Data ------
# Dropping features that serve no use to us
print("Before", training_set.shape, testing_set.shape, combine[0].shape, combine[1].shape)

# Dropping from both testing and training sets
training_set = training_set.drop(['Ticket', 'Cabin'], axis=1)
testing_set = testing_set.drop(['Ticket', 'Cabin'], axis=1)
combine = [training_set, testing_set]

print("After", training_set.shape, testing_set.shape, combine[0].shape, combine[1].shape)
print('\n')

# Creating new features from exisiting 
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print(pd.crosstab(training_set['Title'], training_set['Sex']))
print('\n')

# Replacing titles with a more common name or classifying as rare
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
print(training_set[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
print('\n')

# Convert the categorical titles to ordinal and map them into the datasets 
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

print(training_set.head())
print('\n')

# Drop the name and passengerId from the training set...
# And the name from the testing set (TODO: POSSIBLY REMOVE THE PASSENGER ID FROM THE TESTING SET)
training_set = training_set.drop(['Name', 'PassengerId'], axis=1)
testing_set = testing_set.drop(['Name'], axis=1)
combine = [training_set, testing_set]
print(training_set.shape, testing_set.shape)
print('\n')

# Converting sex from categorical to numerical feature
# This is required by most model algorithms
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

print(training_set.head())
print('\n')

# Completing a numerical continuous feature
# Correlation between Age, Sex, and PClass
grid = sns.FacetGrid(training_set, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

# will contain guessed age values for each sex and pclass
guess_ages = np.zeros((2,3))
for dataset in combine:
    # for each gender
    for i in range(0, 2):
        # for each pclass
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

print(training_set.head(6))
print('\n')

# Create age bands and determine correlation with survival 
training_set['AgeBand'] = pd.cut(training_set['Age'], 5)
print(training_set[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))
print('\n')

# Replacing age with ordinal age bands
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']

print(training_set.head())
print('\n')

# Now drop the age band feature 
training_set = training_set.drop(['AgeBand'], axis=1)
combine = [training_set, testing_set]
print(training_set.head())
print('\n')

# Create new feature combining Parch and SibSp => FamilySize
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

print(training_set[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('\n')

# Create another feature called IsAlone
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print(training_set[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
print('\n')

# Dropping Parch, SibSp, and FamilySize features in favor of IsAlone
training_set = training_set.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
testing_set = testing_set.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [training_set, testing_set]

print(training_set.head())
print('\n')

# Creating an artifical feature combining PClass and Age (TODO: Not sure why we would do this or if it even helps)
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

training_set.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

# Completing the Embarked categorical feature 
freq_port = training_set.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
print(training_set[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('\n')

# Converting the Embarked categorical feature to numeric
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

print(training_set.head())
print('\n')

# Completing the Fare attribute (TODO: WHY DO WE DO THIS FOR THE TEST SET?)
testing_set['Fare'].fillna(testing_set['Fare'].dropna().median(), inplace=True)
print(testing_set.head())
print('\n')

# Create the fareband feature 
training_set['FareBand'] = pd.qcut(training_set['Fare'], 4)
print(training_set[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))
print('\n')

# Convert the fareband feature to ordinal
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

training_set = training_set.drop(['FareBand'], axis=1)
combine = [training_set, testing_set]
    
print(training_set.head(10))
print('\n')
print(testing_set.head(10))
print('\n')
# --------------------------------------------------------------

# ------ Model, predict and solve ------
X_train = training_set.drop("Survived", axis=1)
Y_train = training_set["Survived"]
X_test  = testing_set.drop("PassengerId", axis=1).copy()
print(X_train.shape, Y_train.shape, X_test.shape)
print('\n')

# Logistic Regression 
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(acc_log)
print('\n')

# Modeling correlation (TODO: Note that this is different values)
coeff_df = pd.DataFrame(training_set.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

print(coeff_df.sort_values(by='Correlation', ascending=False))
print('\n')

# Support Vector Machines (TODO: Note that this is a different value)
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print(acc_svc)
print('\n')

# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print(acc_knn)
print('\n')

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print(acc_gaussian)
print('\n')

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print(acc_perceptron)
print('\n')


# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print(acc_linear_svc)
print('\n')


# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print(acc_sgd)
print('\n')


# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print(acc_decision_tree)
print('\n')


# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)
print('\n')


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))