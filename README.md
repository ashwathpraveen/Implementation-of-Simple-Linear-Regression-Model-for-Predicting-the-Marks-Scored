# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. mport the standard libraries

2.Set variables for assigning dataset values

3.Import linear regression from sklearnr

4.Assign the points for representing the graph

5.Predict the regressio for makes by using the representation of the graph

6.Compare the graphs and hence we obtained the linear regression for the given datas


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Ashwath p
RegisterNumber: 212224220012
*/
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(*X)
Y=df.iloc[:,1].values
print(*Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
print(*Y_pred)
Y_test
print(*Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test, y_test, color="blue")
plt.plot(x_test, reg.predict(x_test), color="green")
plt.title('Testing set (Hours vs Scores)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mae = mean_absolute_error(y_test, Y_pred)
mse = mean_squared_error(y_test, Y_pred)
rmse = np.sqrt(mse)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

```

## Output:
<img width="1681" height="671" alt="Screenshot 2026-02-02 212318" src="https://github.com/user-attachments/assets/db445ab4-fe0d-49e7-be89-ef696d0e045b" />
<img width="855" height="638" alt="Screenshot 2026-02-02 212330" src="https://github.com/user-attachments/assets/a3dbab4e-76ad-4799-9ee3-00f6a52c1a21" />




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
