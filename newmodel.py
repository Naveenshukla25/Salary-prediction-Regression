from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import  pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error,r2_score

#load in data 
df = pd.read_csv('Salary_dataset.csv')
#initialize
x = np.array(df['YearsExperience']).reshape(-1,1)
y = np.array(df['Salary']).reshape(-1,1) 
print(x.shape,y.shape)

#creating the model
x_train,x_test,y_train,y_test = train_test_split(x,y)

model = LinearRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test) 

#Evaluate model prediction
r2 = r2_score(y_test,y_pred) 

print("=== Linear Regression Model Performance ===")
print(f"R-squared (R2) Score: {r2:.4f}")

#plotting
plt.scatter(x , y , c='r')
plt.plot(x,y,c='b')
plt.show()