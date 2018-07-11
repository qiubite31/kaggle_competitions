"""
Data Wrangling
=================== =========================== ======================= 
Feature             Missing Value                Transform
=================== =========================== =======================
Embark              依Survived補最多人出發的Port 'C': 0, 'Q': 1, 'S': 2
Fare                依Pclass的平均Fare補值
Age                 取全部age的median
Sex                                             'female': 0, 'male': 1                                           
Family                                          SibSp + Parch
=================== =========================== =======================
Training
=================== =========================== 
Model               Public Score
Decision Tree       0.69378
Random Forest       0.71770
SVM                 0.60766

"""
import os
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt

train_df = pd.read_csv('dataset/train.csv', header=0)
test_df = pd.read_csv('dataset/test.csv', header=0)

# Data Wrangling
# Embarked: train資料有缺值，須轉成numeric
# 統計Survived=1的各Port出發數量並畫bar chart
plt.figure()
s_cnt = len(train_df[ (train_df['Survived'] == 1) & (train_df['Embarked'] == 'S') ])
c_cnt = len(train_df[ (train_df['Survived'] == 1) & (train_df['Embarked'] == 'C') ])
q_cnt = len(train_df[ (train_df['Survived'] == 1) & (train_df['Embarked'] == 'Q') ])
embarK_plot_df = pd.DataFrame(np.array([s_cnt, c_cnt, q_cnt]), index=['S', 'C', 'Q'])
embarK_plot_df.plot.bar()
# plt.show()
# 缺值的乘客Survived=1，填入Survived=1最多人出發的Southampton
train_df['Embarked'] = train_df['Embarked'].fillna('S')
# 新增新的Port欄位，將Embarked轉為數值
train_df['Port'] = train_df['Embarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)
test_df['Port'] = train_df['Embarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)

# Fare: test資料有缺值
# 計算Pclass=1/2/3的平均Fare
fare_pclass1 = test_df[test_df['Pclass'] == 1]['Fare'].mean()
fare_pclass2 = test_df[test_df['Pclass'] == 2]['Fare'].mean()
fare_pclass3 = test_df[test_df['Pclass'] == 3]['Fare'].mean()
# 依據缺值的pclass填入對應平均值
test_df.loc[test_df['Fare'].isnull() & test_df['Pclass'] == 1, 'Fare'] = fare_pclass1
test_df.loc[test_df['Fare'].isnull() & test_df['Pclass'] == 2, 'Fare'] = fare_pclass2
test_df.loc[test_df['Fare'].isnull() & test_df['Pclass'] == 3, 'Fare'] = fare_pclass3

# Age: train/test資料有缺值
train_df['AgeFill'] = train_df['Age']
train_age_median = train_df['Age'].dropna().median()
train_df.loc[train_df.Age.isnull(), 'AgeFill'] = train_age_median
test_df['AgeFill'] = test_df['Age']
test_age_median = test_df['Age'].dropna().median()
test_df.loc[test_df.Age.isnull(), 'AgeFill'] = test_age_median

# Sex: 須轉成numeric
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
# test_df['Gender'] = test_df['Sex'].map( lambda x: 0 if x == 'female' else 1 )
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# SibSp/Parch: 相加擷取成family
train_df['Family'] = train_df['SibSp'] + train_df['Parch']
test_df['Family'] = test_df['SibSp'] + test_df['Parch']

x_train = train_df.drop(['PassengerId', 'Survived', 'Name', 'Age', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)
y_train = train_df['Survived']
# x_test = test_df.drop(['PassengerId', 'Name', 'Age', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)
# x_train = train_df.drop(['PassengerId', 'Name', 'Age', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)

# print(x_test[(x_train['Pclass'].notnull()) & (x_test['Fare'].notnull()) & (x_test['Port'].notnull()) &
#       (x_test['Gender'].notnull()) & (x_test['Family'].notnull()) & (x_test['AgeFill'].notnull())])

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

skf = StratifiedKFold(n_splits=10)
# skf.get_n_splits(x_train, y_train)
for train_idx, test_idx in skf.split(x_train, y_train):
    # print("TRAIN:", train_idx, "TEST:", test_idx)
    # new_train_df = x_train.ix[train_idx]
    # new_test_df = x_train.ix[test_idx]
    feature_train = x_train.ix[train_idx]
    class_train = y_train.ix[train_idx]
    feature_test = x_train.ix[test_idx]
    class_test = y_train.ix[test_idx]

    parameters = {'kernel': ('rbf', 'linear'),
                  'C': [32.0/16.0, 1.0/16.0, 0.25, 0.5, 1, 2, 4, 16, 32],
                  'gamma': [32.0/16.0, 1.0/16.0, 0.25, 0.5, 1, 2, 4, 16, 32]}
    svr = SVC()
    clf = GridSearchCV(svr, parameters)



    # clf = SVC(kernel='rbf', C=10000, gamma=0.001)
    clf.fit(feature_train, class_train)
    print(clf.best_estimator_)
    pred = clf.predict(feature_test)
    acc = accuracy_score(pred, class_test)
    print(acc) 


# new_train_df = train_df.iloc[:len(train_df.index)-100]
# new_test_df = train_df.iloc[len(train_df.index)-100:]




# x_train = new_train_df.drop(['PassengerId', 'Survived', 'Name', 'Age', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)
# y_train = new_train_df['Survived']
# x_test = new_test_df.drop(['PassengerId', 'Survived', 'Name', 'Age', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)
# y_test = new_test_df['Survived']

# linear = 0.77(c=default), 0.86(c=10000) 
# rbf = 0.66(c=10000), 0.76(c=default)

# clf = SVC(kernel='rbf', C=10000, gamma=0.001)
# clf.fit(x_train, y_train)
# pred = clf.predict(x_test)
# acc = accuracy_score(pred, y_test)
# print(acc)
# report = classification_report(pred, y_test)
# print(report)
# print(clf.support_vectors_)

'''
parameters = {'C':[1e3, 5e3, 1e4, 5e4, 1e5, 1, 10, 100, 1000, 10000], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
svr = SVC(kernel='rbf')
clf = GridSearchCV(svr, parameters)
clf = clf.fit(x_train, y_train)

predict = clf.predict(x_test) 

acc = accuracy_score(predict, y_test)
print(acc)
report = classification_report(predict, y_test)
print(report)
'''


output_df = pd.DataFrame(test_df['PassengerId'])
output_df['Survived'] = pd.DataFrame(np.array(pred))
print(output_df.info())
print(output_df)
output_df.to_csv('dataset/output.csv', index=False)

