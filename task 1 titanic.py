#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc


# In[2]:


td = pd.read_csv("Titanic-Dataset.csv")


# In[3]:


td.head()


# In[4]:


td.tail()


# In[6]:


td.dropna(inplace=True)
X = td[['Pclass', 'Sex', 'Age', 'Fare']]
X = pd.get_dummies(X, columns=['Sex'], drop_first=True)
Y = td['Survived']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[7]:


model = RandomForestClassifier(random_state=42)
model.fit(X_train, Y_train)


# In[8]:


person_features = [[3, 30, 50, 0]] 
person_Survival_predictions = model.predict(person_features)
if person_Survival_predictions[0] == 1:
    print("The person is predicted to have survived.")
else:
    print("The person is predicted to have not survived.")


# In[9]:


predictions = model.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy:", accuracy)


# In[10]:


Y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




