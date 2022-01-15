import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\DEVANAND R\Desktop\Data Science Assignments\11. K-Nearest Neighbour\Zoo.csv')

df.isnull().sum() #checking null values

#dropping animal name

df = df.drop('animal name',  axis = 1)

X = df.drop('type', axis = 1)

Y = df['type']

#all values are in numerical format

from sklearn.model_selection import train_test_split

x, x_test, y, y_test = train_test_split(X, Y, test_size = 0.3 )

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

train_acc = []
test_acc = []


for i in range(1,51):
    model = KNeighborsClassifier(n_neighbors = i)
    model.fit(x,y)
    train_acc.append(model.score(x,y))
    pred = model.predict(x_test)
    test_acc.append(accuracy_score(y_test,pred))
    
import matplotlib.pyplot as plt

plt.plot(range(1,51),[i for i in train_acc],"ro-") #train accuracy plot

plt.plot(range(1,51),[i for i in test_acc],"ro-") #test accuracy plot

#from these plots , the optimu k value = 3

#final model

model = KNeighborsClassifier(n_neighbors = 3)
model.fit(x,y)
model.score(x,y) #test accuracy  = 0.95
pred = model.predict(x_test)
accuracy_score(y_test, pred) # test accuracy =0.93

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, pred)
