import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing

df= pd.read_csv('hcvdat0.csv',sep=',')
myData = df.drop(df[df.Category.eq('0s=suspect Blood Donor')].sample(frac=1).index);


le = preprocessing.LabelEncoder()
category = le.fit_transform(myData.Category)
sex = le.fit_transform(myData.Sex)

MeanValues = {
          'ALB':myData.ALB.mean() , 
          'ALP': myData.ALP.mean(), 
          'ALT': myData.ALT.mean(), 
          'CHOL': myData.CHOL.mean(),
          'PROT': myData.PROT.mean(),           
         }
myData = myData.fillna(value = MeanValues)

X = myData.drop(['Category','Sex'], axis=1)
y = myData['Category']
X = pd.get_dummies(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)

##############################################################
from sklearn.neighbors import KNeighborsClassifier

modelKNN = KNeighborsClassifier(n_neighbors=7)
modelKNN.fit(X_train, Y_train)

y_pred= modelKNN.predict(X_test)
conf_mat=confusion_matrix(Y_test, y_pred)

accuracy = modelKNN.score(X_test, Y_test)
print(f"Accuracy of KNN: {accuracy:.2f}")
print(conf_mat)
print(classification_report(Y_test, y_pred))

##############################################################
from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, Y_train)
y_pred1 = svm.predict(X_test)

accuracy = svm.score(X_test, Y_test)
conf_mat=(confusion_matrix(Y_test, y_pred1))
print(conf_mat)
print(f"Accuracy of SVM: {accuracy:.2f}")
print(classification_report(Y_test, y_pred1))

###################################################
from sklearn.tree import DecisionTreeClassifier

modelKA = DecisionTreeClassifier()
modelKA.fit(X, y)

y_pred2 = modelKA.predict(X_test)
conf_mat=confusion_matrix(Y_test, y_pred2)
accuracy = modelKA.score(X_test, Y_test)
print(f"Accuracy of KA: {accuracy:.2f}")
print(conf_mat)
print(classification_report(Y_test, y_pred2))

# Karar ağacını görselleştirme
from sklearn.tree import plot_tree
plot_tree(modelKA)
