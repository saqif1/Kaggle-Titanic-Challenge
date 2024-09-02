import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

### Load CSVs
test_set = pd.read_csv("test.csv", index_col=["PassengerId"]).drop(columns=["Cabin", "Ticket", "Name"])
train_set = pd.read_csv("train.csv", index_col=["PassengerId"]).drop(columns=["Cabin", "Ticket", "Name"]).dropna()

### Pre-processing
# Convert categorical to numerical
test_set['Sex'] = test_set['Sex'].map({"male":1, "female":0})
test_set['Embarked'] = test_set['Embarked'].map({"S":0, "C":1, "Q": 2})
train_set['Sex'] = train_set['Sex'].map({"male":1, "female":0})
train_set['Embarked'] = train_set['Embarked'].map({"S":0, "C":1, "Q": 2})

# Feature Scaling
scaler = MinMaxScaler()
test_set['Age'] = scaler.fit_transform(test_set.Age.values.reshape(-1,1)) #(-1,1) will reshape in such a way that theres only 1 col
test_set['Fare'] = scaler.fit_transform(test_set.Fare.values.reshape(-1,1))
train_set['Age'] = scaler.fit_transform(train_set.Age.values.reshape(-1,1))
train_set['Fare'] = scaler.fit_transform(train_set.Fare.values.reshape(-1,1))

# train test splitting
X_train, X_test, y_train, y_test = train_test_split(train_set.drop(columns=['Survived']), train_set.loc[:,'Survived'], test_size=0.2, random_state=0)

# Logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model's accuracy is: {accuracy}")

# Submission csv
test_set[['Age', 'Fare']] = test_set[['Age', 'Fare']].fillna(test_set[['Age', 'Fare']].median())
test_set_pred = model.predict(test_set)
test_set['Survived'] = test_set_pred
#test_set.Survived.to_csv('Titanic_submission.csv')



