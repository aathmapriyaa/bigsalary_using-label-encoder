import pandas as pd 
import matplotlib.pyplot as pt
import sklearn.neighbors as knn 
from sklearn.preprocessing import LabelEncoder
mydata=pd.read_csv("big_salary_data.csv") 
le = LabelEncoder() 
mydata["education_qualification_encoded"] = le.fit_transform(mydata[["education_qualification"]])
x = mydata[["education_qualification_encoded"]]
y = mydata["salary"]
model=knn.KNeighborsRegressor(n_neighbors=3)
model.fit(x,y)
print(model.predict([[2]]))  # Assuming '2' is the encoded value for education qualification
