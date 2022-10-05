# ----------------------------
# Code for improving the data
# preprocessing step for the 
# kaggle titanic competition
# Adapted from https://www.kaggle.com/code/preejababu/titanic-data-science-solutions#Acquire-data
# -----------------------------
# data analysis and wrangling
import pandas as pd

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

training_path = 'input/train.csv'
testing_path = 'input/test.csv'

training_set = pd.read_csv(training_path)
testing_set = pd.read_csv(testing_path)
combine = [training_set, testing_set]

# Data preprocessing
# Dropping ticket and cabin features from training and testing 
training_set = training_set.drop(['Ticket', 'Cabin'], axis=1)
testing_set = testing_set.drop(['Ticket', 'Cabin'], axis=1)
combine = [training_set, testing_set]

# Generating a new feature (title) from name
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print(training_set[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

# Converting titles to ordinal
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# Dropping name from testing and training 
# and dropping passengerId from testing 
training_set = training_set.drop(['Name', 'PassengerId'], axis=1)
testing_set = testing_set.drop(['Name'], axis=1)
combine = [training_set, testing_set]

# Binarization of sex
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# Filling gaps in Age
print("\nBefore inserting age")
print(training_set.head(10))

# Interpolating missing age values separately 
training_set['Age'] = training_set['Age'].interpolate()
testing_set['Age'] = training_set['Age'].interpolate()

print("After inserting age")
print(training_set.head(10))

# Combining existing features parch and sibsp
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

training_set = training_set.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
testing_set = testing_set.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [training_set, testing_set]
    
# Completing Embarked and turning it to numeric
freq_port = training_set.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
# Completing fare in testing set then making it ordinal
testing_set['Fare'].fillna(testing_set['Fare'].dropna().median(), inplace=True)  

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

combine = [training_set, testing_set]

print("\nFinal Working test and training sets:")
print('\n', training_set.head(10))
print('\n', testing_set.head(10))

# # ------ Model, predict and solve ------
X_train = training_set.drop("Survived", axis=1)
Y_train = training_set["Survived"]
X_test  = testing_set.drop("PassengerId", axis=1).copy()

# Logistic Regression 
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

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

submission = pd.DataFrame({
        "PassengerId": testing_set["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission_improved.csv', index=False)