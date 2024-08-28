from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import warnings
import pickle
import os

warnings.filterwarnings('ignore')
df_description = pd.read_csv('./datasets/description.csv')
df_diets = pd.read_csv('./datasets/diets.csv')
df_medications = pd.read_csv('./datasets/medications.csv')
df_precautions = pd.read_csv('./datasets/precautions_df.csv')
df_symptoms = pd.read_csv('./datasets/symtoms_df.csv')
df_training = pd.read_csv('./datasets/Training.csv')
df_workout = pd.read_csv('./datasets/workout_df.csv')

print(df_training.head())
print(df_training.shape)
print(df_training['prognosis'].unique())
X = df_training.drop('prognosis',axis=1)
y = df_training['prognosis']
le = LabelEncoder()
le.fit(y)
y = le.fit_transform(y)
print(y)

X_test,X_train,y_test,y_train = train_test_split(X,y,test_size=0.3,random_state=42)
models ={
    'SVC':SVC(kernel='linear'),
    'RandomForestClassifier':RandomForestClassifier(random_state=42,n_estimators=100),
    'GradientBoostingClassifier':GradientBoostingClassifier(random_state=42,n_estimators=100),
    'KNeighborsClassifier':KNeighborsClassifier(n_neighbors=5),
    'MultinomialNB':MultinomialNB()
}
for model_name ,models in models.items():
    print(model_name)
    models.fit(X_train,y_train)
    y_pred = models.predict(X_test)
    print(accuracy_score(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))

pickle.dump(models,open('model.pkl','wb'))

