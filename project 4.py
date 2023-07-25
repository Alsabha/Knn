import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler

#Load the wine dataset
data=load_wine()
x,y=data.data,data.target

#split the data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#scale the data
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

#create the KNN classifier
knn=KNeighborsClassifier()

#hyperparameter tuning with GridSearchy
param_grid={'n_neighbors':np.arange(1,21)}
grid_search=GridSearchCV(knn,param_grid,cv=5)
grid_search.fit(x_train_scaled,y_train)

best_k=grid_search.best_params_['n_neighbors']
print("Best k value:",best_k)

knn=KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train_scaled,y_train)

y_pred=knn.predict(x_test_scaled)

accuracy=accuracy_score(y_test,y_pred)
print('Accuracy:',accuracy)

target_names=data.target_names
print("Classification Report:")
print(classification_report(y_test,y_pred,target_names=target_names))

plt.figure(figsize=(6,4))
sns.countplot(x=y,palette='coolwarm')
plt.xticks(ticks=np.unique(y),labels=target_names,rotation=45)
plt.xlabel('Class')
plt.ylabel('count')
plt.title('Class Distribution')
plt.show()

conf_matrix=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='coolwarm',xticklabels=target_names)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()

