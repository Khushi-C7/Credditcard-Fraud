import cartopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import scipy
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest

from sklearn.svm import OneClassSVM
from google.colab import drive

df= pd.read_csv("/content/creditcard.csv")
df.head()

df.shape
df.isnull().sum()
fraud_check = pd.value_counts(df['Class'], sort = True)
fraud_check.plot(kind = 'bar', rot=0, color= 'r')
plt.title("Normal and Fraud Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.xticks(range(2))
plt.show()

fraud_people = df[df['Class']==1]
normal_people= df[df['Class']==0]

fraud_people.shape
normal_people.shape

fraud_people['Amount'].describe()
fraud_people['Amount'].describe()

graph, (plot1, plot2) = plt.subplots(2,1,sharex= True)
graph.suptitle('Average amount per class')
bins = 70

plot1.hist(fraud_people['Amount'] , bins = bins)
plot1.set_title('Fraud Amount')

plot2.hist(normal_people['Amount'] , bins = bins)
plot2.set_title('Normal Amount')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.yscale('log')
plt.show();


plt.figure(figsize=(30,30))
g=sns.heatmap(df.corr(),annot=True)

columns = df.columns.tolist()
# Making our Independent Features
columns = [var for var in columns if var not in ["Class"]]
# Making our Dependent Variable
target = "Class"
x= df[columns]
y= df[target]

x.shape
y.shape
x.head()
y.head()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# Impute missing values using SimpleImputer before fitting the model
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')  # Replace NaN with the mean of the column
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test) # Also impute x_test for consistency

iso_forest= IsolationForest(n_estimators=100, max_samples=len(x_train),random_state=0, verbose=0)
iso_forest.fit(x_train,y_train)
ypred= iso_forest.predict(x_test)
ypred


ypred[ypred == 1] = 0
ypred[ypred == -1] = 1

print(accuracy_score(y_test,ypred))
print(classification_report(y_test,ypred))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, ypred)

n_errors = (ypred != y_test).sum()
print("Isolation Forest have {} errors.".format(n_errors))
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')  # Replace NaN with the mean of the column
x_train_imputed = imputer.fit_transform(x_train)

# Now fit the model with the imputed data
svm = OneClassSVM(kernel='rbf', degree=3, gamma=0.1, nu=0.05, max_iter=-1)
svm.fit(x_train_imputed)

ypred1= svm.predict(x_test)
ypred1[ypred1 == 1] = 0
ypred1[ypred1 == -1] = 1

print(accuracy_score(y_test,ypred))
print(classification_report(y_test,ypred))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, ypred)
n_errors = (ypred1 != y_test).sum()
print("SVM have {} errors.".format(n_errors))


df= pd.read_csv("creditcard.csv")
df.head()
from pycaret.classification import *
model= setup(data= df, target='Class')
compare_models()
random_forest= create_model('rf')
random_forest

tuned_model= tune_model('random_forest')
pred_holdout = predict_model(random_forest,data= x_test)
pred_holdout



