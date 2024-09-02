import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

### Load train dataset
df1 = pd.read_csv("train.csv", index_col=['PassengerId'])

### Data Engineering/ Pre-processing/ Cleaning
# Extracting titles from Name col
df1['Title'] = df1['Name'].map(lambda name: name.rsplit(',')[1].rsplit('.')[0].strip())
Title_Dictionary = {"Capt": "Officer","Col": "Officer","Major": "Officer","Jonkheer": "Royalty","Don": "Royalty",
                    "Sir" : "Royalty", "Dr": "Officer","Rev": "Officer","the Countess":"Royalty","Mme": "Mrs",
                    "Mlle": "Miss","Ms": "Mrs","Mr" : "Mr", "Mrs" : "Mrs","Miss" : "Miss","Master" : "Master","Lady" : "Royalty"}
df1['Title'] = df1['Title'].map(Title_Dictionary)

# Convert Title column to numerical data
Title_rank_dict = {}
count = 0
for title in df1.Title.unique():
    Title_rank_dict[title] = count
    count += 1
print(f"Title_dict :: {Title_rank_dict}")
df1['Title'] = df1.Title.map(Title_rank_dict)

# Change Sex column to numeric data {'M':1, 'F':0}
df1.Sex = df1.Sex.map({'male':1,'female':0})

# Filling NaN in Age column using Median values by gender
male_median_age = df1.loc[df1.Sex==1,['Age']].median()
df1.loc[df1.Sex==1,['Age']] = df1.loc[df1.Sex==1,['Age']].fillna(male_median_age)

female_median_age = df1.loc[df1.Sex==0,['Age']].median()
df1.loc[df1.Sex==0,['Age']] = df1.loc[df1.Sex==0,['Age']].fillna(female_median_age)

# Change Embarked port to numerical data
Embarked_map = {'S':0, 'C':1, 'Q':2, 'nan':np.NaN}
df1['Embarked'] = df1.Embarked.map(Embarked_map)

# Port of embarkation has 2 NaN however, we cannot just fillna with mean/median due to categorical data and since number is small we dropna
df1.dropna(subset=['Embarked'], inplace=True)

# Dropping unnecessary columns
df1 = df1.drop(columns=['Name', 'Ticket', 'Cabin'])

### Feature Scaling
# Normalising the Age (non-categorical) column as high values can distort the model
scaler = MinMaxScaler()
df1['Age'] = scaler.fit_transform(df1['Age'].values.reshape(-1, 1))

# Normalising the Fare (non-categorical) column to prevent distortion
df1['Fare'] = scaler.fit_transform(df1['Fare'].values.reshape(-1, 1))

### Data Modelling
# Train test split
X_train, X_test, y_train, y_test = train_test_split(df1.iloc[:,1:], df1.iloc[:,0], test_size=0.2, random_state=0, stratify=df1.Survived)

model = LogisticRegression()
model.fit(X_train, y_train)

# Get y_pred
y_pred = model.predict(X_test)

# Compare accuracy of y_pred to y_test
acc = accuracy_score(y_test, y_pred)
print(f"Model's accuracy score :: {acc}")

# Display confusion metrix {0: False, 1: True}
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix :: {cm}")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel() # .ravel() flattens the matrix
print(f" tn :: {tn}\n fp :: {fp}\n fn :: {fn}\n tp :: {tp}")
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix')
#plt.show()

### Clean test data for prediction/submission to Kaggle
df2 = pd.read_csv('test.csv', index_col='PassengerId')

df2['Title'] = df2['Name'].map(lambda name: name.rsplit(',')[1].rsplit('.')[0].strip())
df2['Title'] = df2['Title'].map(Title_Dictionary)
df2['Title'] = df2.Title.map(Title_rank_dict)

df2.Sex = df2.Sex.map({'male':1,'female':0})

male_median_age = df2.loc[df2.Sex==1,['Age']].median()
df2.loc[df2.Sex==1,['Age']] = df2.loc[df2.Sex==1,['Age']].fillna(male_median_age)
female_median_age = df2.loc[df2.Sex==0,['Age']].median()
df2.loc[df2.Sex==0,['Age']] = df2.loc[df2.Sex==0,['Age']].fillna(female_median_age)

df2['Embarked'] = df2.Embarked.map(Embarked_map)

df2 = df2.drop(columns=['Name', 'Ticket', 'Cabin'])

# Retrieve those columns with na values for investigation
for col in df2.columns:
    if len(df2.loc[df2[col].isna() == True, :]) != 0:
        print(df2.loc[df2[col].isna() == True, :])


df2['Age'] = scaler.fit_transform(df2['Age'].values.reshape(-1, 1))
df2['Fare'] = scaler.fit_transform(df2['Fare'].values.reshape(-1, 1))

# For the 1st one put in a median value for missing Fare
df2.loc[1044,'Fare'] = df2.Fare.median()

# For the 2nd one since she is 39 yrs old female put in as {Mrs:1} as title
df2.loc[1306,'Title'] = 1

# Check arrangement before predict
col_arrangement = np.unique(X_test.columns == df2.columns)
print(f"Column arrangement is similar? :: {col_arrangement}")

# Model's prediction submission
y_submission = model.predict(df2)
df2['Survived'] = y_submission
print(df2.Survived)
df2.loc[:,'Survived'].to_csv('Titanic_submission.csv')






