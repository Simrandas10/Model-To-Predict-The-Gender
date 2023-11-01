import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('voice.csv')

df.head()

df.isnull().sum()

df.dtypes

#create new data frame with count
new_df=df['label'].value_counts().rename_axis('Male_or_Female').reset_index(name='Count')
new_df.head()

#set labels and values
y_labels=new_df.Male_or_Female
y_values=new_df.Count

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.axis('equal')
ax.pie(y_values, labels=y_labels, autopct='%1.2f%%')
plt.show()

df_copy=df.copy()

df_copy.label.replace({'female':0,'male':1},inplace=True)

df_copy.label.unique()

df_copy.dtypes

%matplotlib inline

X=df_copy.iloc[:,:-1]
Y=df_copy['label']
X

Y

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=10)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

def mscore(model):
    print('Training Score',model.score(X_train,Y_train))
    print('Testing Score',model.score(X_test,Y_test))

def gen_matrix(Y_test,Y_pred):
    cm  = confusion_matrix(Y_test,Y_pred)
    print(cm)
    print(classification_report(Y_test,Y_pred))
    print('Accyracy Score',accuracy_score(Y_test,Y_pred))

m1=DecisionTreeClassifier(criterion='gini',max_depth=4,min_samples_split=12)
m1.fit(X_train,Y_train)

mscore(m1)

ypred_m1 = m1.predict(X_test)
print(ypred_m1)

gen_matrix(Y_test,ypred_m1)

from sklearn.tree import plot_tree

fn=X_train.columns
cn=['0','1']
print(fn)
print(cn)

plt.figure(figsize=(22,7))
plot_tree(m2,feature_names=fn , class_names=cn,filled=True)
plt.show()

m2 = LogisticRegression(max_iter=1000)
m2.fit(X_train,Y_train)

mscore(m2)

ypred_m2 = m2.predict(X_test)
print(ypred_m2)

gen_matrix(Y_test,ypred_m2)

m3 = KNeighborsClassifier(n_neighbors=9)
m3.fit(X_train,Y_train)

mscore(m3)

ypred_m3 = m3.predict(X_test)
print(ypred_m3)

gen_matrix(Y_test,ypred_m3)

m4 = SVC(kernel='linear',C=10) 
m4.fit(X_train,Y_train)

mscore(m4)

ypred_m4 = m4.predict(X_test)
print(ypred_m4)

gen_matrix(Y_test,ypred_m4)

m5 = RandomForestClassifier(n_estimators=30)
m5.fit(X_train,Y_train)

mscore(m5)

ypred_m5 = m5.predict(X_test)
print(ypred_m5)

gen_matrix(Y_test,ypred_m5)
