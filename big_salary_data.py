import pandas as pd 
import matplotlib.pyplot as pt
import sklearn.linear_model as lm
from sklearn.preprocessing import LabelEncoder
mydata = pd.read_csv("big_salary_data.csv") 
le = LabelEncoder()
mydata["education_qualification_encoded"] = le.fit_transform(mydata[["education_qualification"]]) 
x = mydata[["education_qualification_encoded"]]
y = mydata["salary"] 
model = lm.LinearRegression()
model.fit(x, y)
print("Coefficients:", model.coef_[0])
print("Intercept:", model.intercept_)
print(model.predict([[2]]))  