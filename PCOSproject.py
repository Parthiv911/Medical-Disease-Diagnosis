import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

credit_data = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\MLDATA\PCOS_data.csv")
'''
print(credit_data.describe())
print(credit_data.corr())
'''

features = credit_data.iloc[0:538, 1:39]
target = credit_data.iloc[0:538, 0:1]

print("Data:")
print(features)
print(target)

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3)

target_train = np.ravel(target_train)
target_test = np.ravel(target_test)

model = LogisticRegression(solver='lbfgs', max_iter=10000)
model.fit(feature_train, target_train)

predictions = model.predict(feature_test)
'''
diagonal is correct prediction
        predicted
        0       1
actual 0

       1
'''
cm=confusion_matrix(target_test,predictions)
print("Confusion matrix: \n")

print("               Predicted No     Predicted Yes")
print("Actual No          ",cm[0,0],"              ",cm[0,1])
print("Actual Yes         ",cm[1,0],"              ",cm[1,1])
#total_correct/total
print("\nAccuracy Score: ",accuracy_score(target_test,predictions))