#importing the dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

#Reading the dataset 
df = pd.read_csv('INFY.csv')

#Data column converted into index coloumn 
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df = df.set_index('Date')

#Checking the whole dataset for null vlaues and dtype.
df.info()

#Displaying the first few entries
df.head()

#Ploting the Adj close (because it is the vlue of the volume at the day end) 
#Based on image the given dataset is "non-linear".
plt.figure(figsize=(16,8))
plt.plot(df['Adj Close'], label='Close Price history')

#Taking a single coloumn Adj close
df = df[['Adj Close']] 
df.head()

#Creating the coloumn
days = 1
df['Prediction'] = df[['Adj Close']].shift(-days)
df.tail()

#Creating the X(dependent Variable)
X = np.array(df.drop(['Prediction'],1))
X = X[:-days]

#Creating the y(independent Variable)
y = np.array(df['Prediction'])
y = y[:-days]

#Spliting the Data into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#training the SVR model 
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
svr_rbf.fit(x_train, y_train)

#Printing the r_square_value
r_square_value = svr_rbf.score(x_test, y_test)
print("r_square_value: ", r_square_value)

#Taking the next day dependent value 
x_days_value = np.array(df.drop(['Prediction'],1))[-days:]
print(x_days_value)

#Predicting the next day Adj_close value
svm_pred = svr_rbf.predict(x_days_value)
next_day_value = svm_pred
print(next_day_value)


