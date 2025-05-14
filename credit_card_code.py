import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# reading csv file
credit_card_data =pd.read_csv('creditcard.csv')
# prints first 5 rows of the dataset
credit_card_data.head()
#prints last 5 rows
credit_card_data.tail()
# information about the dataset
credit_card_data.info()
# checking the number of missing values in each column using isnull and sum()
credit_card_data.isnull().sum()
# distributed data number of legit transactions & fraud transactions
credit_card_data['Class'].value_counts()
# separating the data of legit and fraud for detail analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
print(legit.shape)
print(fraud.shape)
# statistical measures of the legit amount in the dataset
legit.Amount.describe()
# statistical measures of the fraud amount in the dataset
fraud.Amount.describe()
# comparing the values of both legit and fraud transactions
credit_card_data.groupby('Class').mean()
# data for legit transactions
legit_sample = legit.sample(n=492)
#Concatenating two Data
new_dataset = pd.concat([legit_sample, fraud], axis=0)
# getting the first 5 rows of new dataset
new_dataset.head()
# getting the last 5 rows of new dataset
new_dataset.tail()
# count  of the new dataset
new_dataset['Class'].value_counts()
# mean value of new dataset
new_dataset.groupby('Class').mean()

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

#model training and logistics regression
model_t = LogisticRegression(max_iter=100000)
# training the Logistic Regression Model with Training Data
model_t.fit(X_train, Y_train)

#Model Evaluation and accuracy check
# accuracy on training data for the model
X_train_predict = model_t.predict(X_train)
final_training_data_accuracy = accuracy_score(X_train_predict, Y_train)
print('Final Accuracy on Training data of the dataset : ', final_training_data_accuracy)


# accuracy on test data for the model
X_test_predict = model_t.predict(X_test)
final_test_data_accuracy = accuracy_score(X_test_predict, Y_test)
print('Final Accuracy score for Test Data of the dataset : ', final_test_data_accuracy)













