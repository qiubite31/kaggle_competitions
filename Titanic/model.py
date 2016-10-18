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
Training Model: Decision Tree
Public Score: 0.69378
"""
import os
import pandas as pd
import numpy as np
from sklearn import tree
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
x_test = test_df.drop(['PassengerId', 'Name', 'Age', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)

print(x_test[(x_train['Pclass'].notnull()) & (x_test['Fare'].notnull()) & (x_test['Port'].notnull()) &
      (x_test['Gender'].notnull()) & (x_test['Family'].notnull()) & (x_test['AgeFill'].notnull())])

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
predict = clf.predict(x_test)
output_df = pd.DataFrame(test_df['PassengerId'])
output_df['Survived'] = pd.DataFrame(np.array(predict))
print(output_df.info())
print(output_df)
output_df.to_csv('dataset/output.csv')
