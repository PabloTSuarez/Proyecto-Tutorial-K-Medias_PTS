from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from pickle import dump


df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv')
df

df1 = df[['Latitude','Longitude','MedInc']]
df1

X_train,X_test = train_test_split(df1,test_size=0.2,random_state=42)
print(X_train.shape)

model = KMeans(n_clusters=6,random_state=42)
model.fit(X_train)

X_train['cluster'] = model.predict(X_train)
X_train.head()

sns.scatterplot(data=X_train,x='Latitude',y='Longitude',hue='cluster',palette='deep')
sns.scatterplot(data=X_train,x='Latitude',y='MedInc',hue='cluster',palette='deep')
sns.scatterplot(data=X_train,x='Longitude',y='MedInc',hue='cluster',palette='deep')

X_test['cluster'] = model.predict(X_test)
X_test.head()

sns.scatterplot(data=X_train,x='Latitude',y='Longitude',hue='cluster',palette='deep',alpha = 0.1)
sns.scatterplot(data=X_test,x='Latitude',y='Longitude',hue='cluster',palette='deep',markers = '*')
plt.show()

model_dt = DecisionTreeClassifier(random_state=42)
model_rf = RandomForestClassifier(random_state=42)

y_train = X_train.cluster
X_train = X_train.drop(['cluster'],axis=1)

model_dt.fit(X_train,y_train)
model_rf.fit(X_train,y_train)

y_test = X_test.cluster
X_test = X_test.drop(['cluster'],axis=1)

y_pred_dt = model_dt.predict(X_test)
y_pred_rf = model_rf.predict(X_test)

as_dt = accuracy_score(y_test,y_pred_dt)
as_rf = accuracy_score(y_test,y_pred_rf)
print(f'La precision del arbol de decision es de {as_dt} y de random forest es {as_rf}')

tree.plot_tree(model_dt,feature_names=X_train.columns,class_names=['0','1','2','3','4','5'],filled=True)
plt.show()

dump(model,open('../models/model_kmean.model','wb'))
dump(model_dt,open('../models/model_dt.model','wb'))
dump(model_rf,open('../models/model_rf.model','wb'))