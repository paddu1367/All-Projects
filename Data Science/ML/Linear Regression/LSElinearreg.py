import pandas as pd
import numpy as np
from numpy import mean,std
from scipy import optimize
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import array as arr
 
#loaind data
df = pd.read_csv(r'iris.csv')
#print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Traning model
x = df.drop('Species', axis=1)
y1 = df['Species']
l = LabelEncoder()
y = l.fit_transform(y1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/2,random_state=1)

#Traing samples for Least Square Estimatior

oness = np.ones(len(x_train))[:, np.newaxis]
A = np.hstack([x_train, oness])

y = y_train[:, np.newaxis]

# Direct least square regression
alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y)
print(alpha)

#Output classification(predction)
y_predect = []
for index, row in x_test.iterrows() :
    z1=alpha[0]*row['sepallength']+alpha[1]*row['sepalwidth']+alpha[2]*row['petallength']+alpha[3]*row['petalwidth']+alpha[4]
    z=[round(num) for num in z1.tolist()]
    y_predect.append(z)

#classification results
print('Accuracy Score for classification: ',accuracy_score(y_test,y_predect))
