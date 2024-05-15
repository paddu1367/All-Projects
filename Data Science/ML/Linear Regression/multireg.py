import pandas as pd
import numpy as np
from numpy import mean,std
import seaborn as sns
import matplotlib.pyplot as plt
 
#loaind data
df = pd.read_csv(r'iris.csv')
#print(df.head())


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score,RepeatedStratifiedKFold

#Traning model

x = df.drop('Species', axis=1)
y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
logmodel = LogisticRegression(multi_class='multinomial', solver='lbfgs' , max_iter=100)
logmodel.fit(x_train, y_train)

#Results verification
predictions = logmodel.predict(x_test)
print('Classification Report of model:\n',classification_report(y_test, predictions))

print('Confusion matrix of model:\n',confusion_matrix(y_test, predictions))
sns.heatmap(pd.DataFrame(confusion_matrix(y_test,predictions)))
plt.show()

print('Accuracy score of model: ',accuracy_score(y_test, predictions))

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(logmodel, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
# report the model performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
