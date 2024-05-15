#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#defining linear Regressin class and functions.
class linear_regression:
    def __init__(self):
        self.weights=0

    def fit(self,x,y):
        q = np.hstack([np.ones(len(x))[:, np.newaxis],x])
        self.weights = (np.dot(np.linalg.inv(q.T.dot(q)),q.T)).dot(y)
        return self.weights
    
    def predict(self,x):
        return np.round(x.dot(self.weights[1:5])+self.weights[0])

def accuracy(x,y):
        count=0
        for i in range(len(x)):
            if x[i]==y[i]:
                count+=1
        return count/len(x)*100

def crossvalidation(x,y,k_folds):
    accuracy_score =[]
    for i in range(0,len(x),k_folds):
        x_train_input = np.vstack((x[0:i,:],x[i+1:,:]))
        x_validation= x[i:i+1,:]
        y_train_input = np.vstack((y[0:i,:],y[i+1:,:]))
        y_validation= y[i:i+1,:]
        model = linear_regression()
        model.fit(x_train_input,y_train_input)
        accuracy_score.append(accuracy(model.predict(x_validation),y_validation))
    return sum(accuracy_score)/len(accuracy_score)

#importing data
data = pd.read_csv("iris.data",names=['x1','x2','x3','x4','class'])
#print(data.head())

#data analysis

#sns.pairplot(data, hue= 'class')
#plt.show()

#linear classification
map = {
    'Iris-setosa' : 0,
    'Iris-versicolor' : 2,
    'Iris-virginica' : 3
}

data.replace(map,inplace=True)

X= np.array(data.drop('class',axis=1))
Y= np.array(data['class'])[:, np.newaxis]

#splitting data
x_train = X[0:135]
x_test = X[135:150]
y_train = Y[0:135]
y_test = Y[135:150] 

li =linear_regression()
print("Coefficients of linear model:\n",li.fit(x_train,y_train))
y_predect = li.predict(x_test)
#classification results
print('\nAccuracy Score for classification: {}%'.format(accuracy(y_test,y_predect)))
#K-folds cross validation results
for i in range(1,11,1):
    print('\nMean Accuracy for {}-folds: {:.2f}%'.format(i,crossvalidation(X,Y,i)))