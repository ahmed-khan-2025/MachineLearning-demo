import pandas as pd
import joblib
model = joblib.load('music-rcmd.joblib')
input_data = pd.DataFrame([[22,0]], columns=['age', 'gender'])

predistion = model.predict(input_data)

print(f" Predicted music genre: {predistion[0]}")