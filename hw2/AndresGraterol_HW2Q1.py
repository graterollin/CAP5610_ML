# ----------------------------
# Andres Graterol 
# CAP5610 - Fall 22
# 4031393
# ----------------------------
# Adapted from https://www.kaggle.com/code/preejababu/titanic-data-science-solutions#Acquire-data 
# with preprocessing based off the warm-up competition assignment 
# ----------------------------
# data analysis and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn import tree

# Models used 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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

# Interpolating missing age values separately 
training_set['Age'] = training_set['Age'].interpolate()
testing_set['Age'] = training_set['Age'].interpolate()
combine = [training_set, testing_set]

# Completing Embarked and turning it to numeric
freq_port = training_set.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#Binarization of sex
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# Combining existing features parch and sibsp
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

training_set = training_set.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
testing_set = testing_set.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [training_set, testing_set]
    
# Completing fare in testing set then making it ordinal
testing_set['Fare'].fillna(testing_set['Fare'].dropna().median(), inplace=True)  

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

#print("\nFinal Working test and training sets:")
#print('\n', training_set.head(10))
#print('\n', testing_set.head(10))

# ----------------------------------------------------------
# Now to handle the decision tree and random forest models 
# ----------------------------------------------------------

# Function for fine-tuning parameters of our model
def fine_tune(model, params):
    grid_search = GridSearchCV(estimator=model,
                               param_grid=params,
                               cv=4, 
                               scoring='accuracy')
    
    return grid_search
    
# Learn and fine-tune a decision tree model with the training data and plot
X_train = training_set.drop("Survived", axis=1)
Y_train = training_set["Survived"]
X_test  = testing_set.drop("PassengerId", axis=1).copy()

# Initialize a Decision Tree 
decision_tree = DecisionTreeClassifier()

# Using GridSearchCV to fine-tune the decision tree 
# Get some sample parameters 
dt_params = {
    'max_depth': [3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
}

dt_grid_search = fine_tune(decision_tree, dt_params)
dt_grid_search.fit(X_train, Y_train)

print("\nBest decision tree parameters:")
print(dt_grid_search.best_estimator_)

# Set the decision tree to the best parameters 
tuned_decision_tree = dt_grid_search.best_estimator_
tuned_decision_tree.fit(X_train, Y_train)
#Y_pred = tuned_decision_tree.predict(X_test)
acc_decision_tree = round(tuned_decision_tree.score(X_train, Y_train) * 100, 2)

# Visualizing the fine-tuned decision tree
tree_plot = plt.figure(figsize=(35,30))
_ = tree.plot_tree(tuned_decision_tree, 
                   feature_names = X_train.columns, 
                   class_names= ['0', '1'], 
                   filled=True)

# Initialize a Random Forest Classifier
random_forest = RandomForestClassifier()

# Using GridSearchCV to fine-tune the random forest  
# Get some sample parameters 
rf_params = {
    'min_samples_split': [3, 5, 10, 20],
    'max_depth': [3, 5, 10, 20],
    'min_samples_leaf': [3, 5, 10, 20]
}

rf_grid_search = fine_tune(random_forest, rf_params)
rf_grid_search.fit(X_train, Y_train)

print("\nBest random forest parameters:")
print(rf_grid_search.best_estimator_)

# Set the random forest model to the best parameters 
tuned_random_forest = rf_grid_search.best_estimator_
tuned_random_forest.fit(X_train, Y_train)
Y_pred = tuned_random_forest.predict(X_test)
acc_random_forest = round(tuned_random_forest.score(X_train, Y_train) * 100, 2)

# These are the fine-tuned model accuracies before cross-validation
print("\nAccuracy Scores for fine-tuned models")
models = pd.DataFrame({
    'Model': ['Random Forest', 'Decision Tree'],
    'Score': [acc_random_forest, acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))

# Now to preform 5 cross fold validation on both models
dt_cv = KFold(n_splits=5, shuffle=True, random_state=1)
dt_scores = cross_val_score(tuned_decision_tree, X_train, Y_train, scoring='accuracy', cv=dt_cv)
cv_acc_decision_tree = round((np.mean(dt_scores)*100), 2)

print("\nCross-Val Scores for decision tree:", dt_scores)

rf_cv = KFold(n_splits=5, shuffle=True, random_state=1)
rf_scores = cross_val_score(tuned_random_forest, X_train, Y_train, scoring='accuracy', cv=rf_cv)
cv_acc_random_forest = round((np.mean(rf_scores)*100), 2)

print("Cross-Val Scores for random forest:", rf_scores)

# These are the fine-tuned model accuracies after cross-validation
print("\nMean Accuracy Scores for cross-validated fine-tuned models")
models = pd.DataFrame({
    'Model': ['Random Forest', 'Decision Tree'],
    'Score': [cv_acc_random_forest, cv_acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))

# submission = pd.DataFrame({
#           "PassengerId": testing_set["PassengerId"],
#           "Survived": Y_pred
#       })
# submission.to_csv('ensemble.csv', index=False)