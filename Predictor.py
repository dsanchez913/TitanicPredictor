import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


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

Y_pred_arr=[]
acc_arr=[]

#Test the logistic regresion algorithm
print()
print('----- Logistic Regression --------')
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()



logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_log = logreg.predict(X_test)

print('Y_pred for Logistic Regression:')
print()
print(Y_pred_log)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

print('Score for logistic regression: '+str(acc_log))

Y_pred_arr.append(Y_pred_log)
acc_arr.append(acc_log)

#Test the SVM algorithm
print()
print('----- Support Vector Machine --------')

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_svm = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

print('Y_pred for SVM:')
print()
print(Y_pred_svm)
print('Score for suport vector machine: '+str(acc_svc))
Y_pred_arr.append(Y_pred_svm)
acc_arr.append(acc_svc)

#Test the k-nearest neighbors algorithm
print()
print('----- k-nearest neighbors --------')

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

print('Y_pred for SVM:')
print()
print(Y_pred_knn)
print('Score for k-nearest neighbors: '+str(acc_knn))
Y_pred_arr.append(Y_pred_knn)
acc_arr.append(acc_knn)

# Test the Gaussian Naive Bayes algorithm
print()
print('----- Gaussian Naive Bayes --------')

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred_gaussian = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

print('Y_pred for Gaussian Naive Bayes:')
print()
print(Y_pred_gaussian)
print('Score for Gaussian Naive Bayes: '+str(acc_gaussian))
Y_pred_arr.append(Y_pred_gaussian)
acc_arr.append(acc_gaussian)

# Test the Perceptron algorithm
print()
print('----- Perceptron --------')
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred_perc = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

print('Y_pred for perceptron:')
print()
print(Y_pred_perc)
print('Score for perceptron: '+str(acc_perceptron))
Y_pred_arr.append(Y_pred_perc)
acc_arr.append(acc_perceptron)

# Test the Linear SVC algorithm
print()
print('----- Linear SVC --------')
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred_linsvc = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

print('Y_pred for linear SVC:')
print()
print(Y_pred_linsvc)
print('Score for perceptron: '+str(acc_linear_svc))
Y_pred_arr.append(Y_pred_linsvc)
acc_arr.append(acc_linear_svc)

# Test the Stochastic Gradient Descent algorithm
print()
print('----- Stochastic Gradient Descent --------')
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred_sgd = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

print('Y_pred for SGD:')
print()
print(Y_pred_sgd)
print('Score for SGD: '+str(acc_sgd))
Y_pred_arr.append(Y_pred_sgd)
acc_arr.append(acc_sgd)

# Test the Decision Tree algorithm
print()
print('----- Decision Tree --------')
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_dt = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

print('Y_pred for Decision Tree:')
print()
print(Y_pred_dt)
print('Score for Decision Tree: '+str(acc_decision_tree))
Y_pred_arr.append(Y_pred_dt)
acc_arr.append(acc_decision_tree)

# Test the Random Forest algorithm
print()
print('----- Random Forest --------')
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred_rf = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print('Y_pred for Random Forest:')
print()
print(Y_pred_rf)
print('Score for Random Forest: '+str(acc_random_forest))
Y_pred_arr.append(Y_pred_rf)
acc_arr.append(acc_random_forest)

#Select the best result
np_acc= np.array(acc_arr)
best=np.where(np_acc==np.amax(np_acc))[0][0]
print('Best index: '+str(best))
Y_best=Y_pred_arr[best]
subm = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_best
    })

#subm.to_csv(r'C:\Users\USER\Dropbox (Personal)\Data Science Projects\TitanicPredictor\submission_LR.csv',index=False)
subm.to_csv('submission_best.csv', index=False)