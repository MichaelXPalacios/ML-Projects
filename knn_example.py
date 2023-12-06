# -*- coding: utf-8 -*-
"""KNN example

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WYTzbtyxCXeNXDREBBXSxswRNYIlb_6o
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
# %matplotlib inline

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# %matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from google.colab import files
uploaded = files.upload()

knn = pd.read_csv('owner.csv')
knn.head(20)

owner_num = {'Owner': 1, 'non-owner': 0}
knn['Ownership'] = knn['Ownership'].map(owner_num)
knn.head(20)

knn.to_csv('owner_cleaned.csv', index = False)

knn["Ownership"].value_counts()

#Scatterplot 3
p =sns.scatterplot(x="Income", y="Lot_Size", hue="Ownership",
data=knn,palette=['orange','dodgerblue'], legend='full')
plt.xlabel('Income')
plt.ylabel('Lot_Size')
p.set(xscale="log")

X = knn.iloc[:, 1:-1].values
    y = knn.iloc[:, 2].values

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=42,)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors = 3)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
print(y_pred)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

error_rate = []

for i in range(1, 7):
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(X_train, y_train)
  pred_i = knn.predict(X_test)
  error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 7), error_rate, color='blue', linestyle='--',markersize=10, markerfacecolor='red', marker='o')
plt.title('K versus Error rate')
plt.xlabel('K')
plt.ylabel('Error rate')

print(error_rate[4])

classifier = KNeighborsClassifier(n_neighbors = 4)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
print(y_pred)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print(y_pred)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

classifier = KNeighborsClassifier(n_neighbors = 6)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print(y_pred)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

classifier = KNeighborsClassifier(n_neighbors = 1)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print(y_pred)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# %matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from google.colab import files
uploaded = files.upload()

lin = pd.read_csv('linear2.csv')
lin.head(20)

# Coomputing X and Y
X = lin['Variable 1'].values
Y = lin['Variable 2'].values

# Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

# Total number of values
n = len(X)

# Using the formula to calculate 'm' and 'c'
numer = 0
denom = 0
for i in range(n):
  numer += (X[i] - mean_x) * (Y[i] - mean_y)
  denom += (X[i] - mean_x) ** 2
  m = numer / denom
  c = mean_y - (m * mean_x)

# Printing coefficients
print("Coefficients")
print(m, c)

# Plotting Values and Regression Line

max_x = np.max(X) + 10
min_x = np.min(X) - 0

# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = c + m * x

# Ploting Line
plt.plot(x, y, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')

plt.xlabel('Variable 1')
plt.ylabel('variable 2')
plt.legend()
plt.show()