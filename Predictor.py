import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression


test_df = pd.read_csv('/Users/dsanchez/Dropbox/Data Science Projects/TitanicPredictor/test.csv')
train_df = pd.read_csv('/Users/dsanchez/Dropbox/Data Science Projects/TitanicPredictor/train.csv')

full_ds=[train_df,test_df]

# first task: understand the data
print('-------These are the values of the CSV columns-------')
print(train_df.columns.values)
print('-------The data frame head-------')
print(train_df.head())
print('-------The data frame descriptors (1)-------')
print(train_df.describe())
print('-------The data frame descriptors (2)-------')
print(train_df.describe(include=['O']))
print('-------DATA ANALYSIS-------')
print('-------Correlation between sex and survival rate-------')
print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('-------Correlation between Passenger class and survival rate-------')
print(train_df[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))


#second task: create a dataframe with only numerical values
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
full_ds=[train_df,test_df]
    #convert sex to a numerical feature
for dataset in full_ds:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    #Create a category set for the Title in the Name column
    #First we must examine tohe Titles
for dataset in full_ds:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
print(pd.crosstab(train_df['Title'], train_df['Sex']))

    #Then we create a set of categories for the Title feature

for dataset in full_ds:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

print(train_df.head())

    #The 5 categories are then mapped into numerical values

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in full_ds:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)


#Now we can drop the Name column and also the passengerId column from the training dataset
#axis=1 indicates that we want to drop a column in the dataset
train_df=train_df.drop(['Name','PassengerId'],axis=1)
test_df=test_df.drop(['Name'],axis=1)
full_ds=[train_df,test_df]

#Completing age features
for i in range(0,2):
    for dataset in full_ds:
        age_sex=dataset.loc[dataset['Sex']==i,'Age']
        length_ds=len(dataset.loc[(dataset.Sex==i)&(dataset.Age.isnull()),'Age'])
        dataset.loc[(dataset.Sex==i)&(dataset.Age.isnull()),'Age']=np.random.uniform(low=age_sex.mean()-age_sex.std(),high=age_sex.mean()+age_sex.std(),size=length_ds).astype(int)


#Fill in the only null value for Fare in the test dataset

test_df['Fare']=test_df['Fare'].fillna(test_df['Fare'].median())

full_ds=[train_df,test_df]


#Create one feature called familiy size in order to merge SibSp and Parch

for dataset in full_ds:
    dataset['FSize']=1+dataset['SibSp']+dataset['Parch']

train_df=train_df.drop(['SibSp','Parch'],axis=1)
test_df=test_df.drop(['SibSp','Parch'],axis=1)
full_ds=[train_df,test_df]

#Create numerical categories for Embarked

freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in full_ds:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


#Test the linear regresion algorithm

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()



logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

print(Y_pred)

subm = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

#subm.to_csv(r'C:\Users\USER\Dropbox (Personal)\Data Science Projects\TitanicPredictor\submission_LR.csv',index=False)
subm.to_csv('/Users/dsanchez/Dropbox/Data Science Projects/TitanicPredictor/submission_LR.csv')