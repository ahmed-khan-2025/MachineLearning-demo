import pandas as pd
df = pd.read_csv('music.csv')

X = df.drop(columns=['genre']) # features
y = df['genre'] # labels

print(y)