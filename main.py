import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


df = pd.read_csv('HCV-Egy-Data.csv',sep=',')
df = df.rename(columns={'Baselinehistological staging': 'stage'})

myData1 = df.drop(df[df.stage.eq(2)].sample(frac=1).index);
myData = myData1.drop(myData1[myData1.stage.eq(3)].sample(frac=1).index);

myData.rename(columns=lambda x: x.strip(), inplace=True)

print(myData.isnull().sum())

X = myData.iloc[:,0:28]
y = myData.iloc[:,28]

print("myData.shape: {} X.shape: {} y.shape: {}".format(myData.shape, X.shape, y.shape))

X = pd.get_dummies(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=70)

float_columns = X_train.select_dtypes(include=['float']).columns

ct = ColumnTransformer([('float_scaler', StandardScaler(), float_columns)], verbose=True)
X_train[float_columns] = ct.fit_transform(X_train[float_columns])
X_test[float_columns] = ct.transform(X_test[float_columns])

#####################################################################
from sklearn.neighbors import KNeighborsClassifier

modelKNN = KNeighborsClassifier(n_neighbors=7)
modelKNN.fit(X_train, Y_train)

y_pred= modelKNN.predict(X_test)
conf_mat=confusion_matrix(Y_test, y_pred)

accuracy = modelKNN.score(X_test, Y_test)
print(f"Accuracy of KNN: {accuracy:.2f}")
print(conf_mat)
print(classification_report(Y_test, y_pred))


####################################################################
from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, Y_train)
y_pred1 = svm.predict(X_test)

accuracy = svm.score(X_test, Y_test)
conf_mat=(confusion_matrix(Y_test, y_pred1))
print(conf_mat)
print(f"Accuracy of SVM: {accuracy:.2f}")
print(classification_report(Y_test, y_pred1))


####################################################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

modelKA = DecisionTreeClassifier()
modelKA.fit(X, y)

y_pred2 = modelKA.predict(X_test)
conf_mat=confusion_matrix(Y_test, y_pred2)
accuracy = modelKA.score(X_test, Y_test)
print(f"Accuracy of KA: {accuracy:.2f}")
print(conf_mat)
print(classification_report(Y_test, y_pred2))

plot_tree(modelKA)
