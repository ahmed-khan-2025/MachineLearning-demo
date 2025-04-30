import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('music.csv')
X = df.drop(columns=['genre']) # features
y = df['genre'] # labels

model = DecisionTreeClassifier()
model.fit(X,y)
predictions = model.predict([[31,1],[21,0]])

print(predictions)