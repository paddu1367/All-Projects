import pandas as pd
import numpy as np
from numpy import mean,std
import seaborn as sns
import matplotlib.pyplot as plt
 
#loaind data
df = pd.read_csv(r'iris.csv')
#print(df.head())


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score,RepeatedStratifiedKFold

#Traning model
mapping = {
    'Iris-setosa' : 1,
    'Iris-versicolor' : 2,
    'Iris-virginica' : 3
}

x = df.drop('sepallength', axis=1).replace(mapping)
y = df['sepallength']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
linmodel = LinearRegression()
linmodel.fit(x_train, y_train)

print('Coefficients: ', linmodel.coef_)




