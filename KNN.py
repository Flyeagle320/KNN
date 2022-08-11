# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 19:03:45 2022

@author: Rakesh
"""

########################Problem 1#########################################
import pandas as pd
import numpy as np

glass  =pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_KNN/glass.csv')

#Normalized function##
def norm_func(i):
    x= ((i-i.min())/(i.max()-i.min()))
    return(x)

#normalized data frame# considering only numerical part of data#
glass_norm = norm_func(glass.iloc[:,0:9])
glass_norm.describe()

x= np.array(glass_norm.iloc[:,:]) # Predictors
y= np.array(glass['Type']) #target#

##splitting the data train and test#
from sklearn.model_selection import train_test_split
x_train,x_test,y_train , y_test= train_test_split(x,y, test_size= 0.2)

##Model building on train data#
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

#prediction on test data#
pred=knn.predict(x_test)
pred

##model evalutaion#
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred))
pd.crosstab(y_test, pred,rownames=['Actual'],colnames=['Predictions'])

##accuracy on train data#
pred_train=knn.predict(x_train)
print(accuracy_score(y_train, pred_train))
pd.crosstab(y_train, pred_train, rownames=['Actual'], colnames=['Predictions'])

##creating empty list variable#
acc = []

##running KN algorithm for 3- 50 nearest neighbours#
##storing accuracy value#
for i in range(3,50,2):
    neigh= KNeighborsClassifier(n_neighbors=i)
    neigh.fit(x_train,y_train)
    train_acc= np.mean(neigh.predict(x_train)==y_train)
    test_acc= np.mean(neigh.predict(x_test)==y_test)
    acc.append([train_acc,test_acc])

##visualization##
import matplotlib.pyplot as plt    
##train accuracy##
plt.plot(np.arange(3, 50,2), [i[0] for i in acc], "ro-")
##test accuracy##
plt.plot(np.arange(3, 50,2), [i[1] for i in acc], "bo-")

#########################problem2############################
import pandas as pd
import numpy as np

zoo= pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_KNN/Zoo.csv')
zoo = zoo.iloc[: , 1:] ##dropping animal name #

##normalized function##
def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return(x)
#normalized data frame# considering only numerical part of data#
zoo_norm=norm_func(zoo.iloc[:,:16])
zoo_norm.describe()

x= np.array(zoo_norm.iloc[:,:]) ##predictor#
y=np.array(zoo['type']) #target#

####splitting the data train and test#
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2)

##modeling for train data#
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

##predicting on test data#
pred= knn.predict(x_test)
pred
##evaluate the model and find accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))
pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predictions'])

##accuracy on train data
pred_train = knn.predict(x_train)
print(accuracy_score(y_train, pred_train))
pd.crosstab(y_train, pred_train, rownames=['Actual'], colnames=['Predictions'])

# creating empty list variable 
acc = []

##running KN algorithm for 3- 50 nearest neighbours#
##storing accuracy value#
for i in range(3,50,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(x_train, y_train)
    train_acc = np.mean(neigh.predict(x_train) == y_train)
    test_acc = np.mean(neigh.predict(x_test) == y_test)
    acc.append([train_acc, test_acc])
    
##visualization##
import matplotlib.pyplot as plt    
##train accuracy##
plt.plot(np.arange(3, 50,2), [i[0] for i in acc], "ro-")
##test accuracy##
plt.plot(np.arange(3, 50,2), [i[1] for i in acc], "bo-")    
#####################################################################    
    
