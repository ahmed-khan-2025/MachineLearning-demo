import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

df = pd.read_csv('music.csv')
X = df.drop(columns=['genre']) # features
y = df['genre'] # labels

model = DecisionTreeClassifier()
model.fit(X,y)

joblib.dump(model,'music-rcmd.joblib')
