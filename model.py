import pandas as pd
from sklearn.linear_model import LinearRegression

ds = pd.read_csv('Salary_Data.csv')
x = ds['YearsExperience'].values.reshape(-1,1)
y = ds['Salary']

mind = LinearRegression()
model = mind.fit(x,y)
print("Welcome to Salary Predictor App \n")
p=input("Enter no. of year :")
s=float(p)

o = model.predict([[s]])
print("Predicted Salary is: ",o ,"\n")