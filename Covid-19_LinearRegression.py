import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

df = pd.read_csv('covid_19_WORLD.csv',parse_dates=['Date'])
df = df.groupby('Date').sum()['Confirmed'].reset_index()
dfc = df[['Confirmed']]
print(dfc)
#predict n days out into the future
forecast_out = 7

#create another column target variable
dfc['Prediction'] = dfc[['Confirmed']].shift(-forecast_out)

#create df into numpy array
x = np.array(dfc.drop(['Prediction'],1))
#remove the last n rows
x = x[:-forecast_out]

#convert df into numpy with NaNs
y = np.array(dfc['Prediction'])

#Get all of y values except last n rows
y = y[:-forecast_out]

#Split data into 80% training and 20% testing
x_train,x_test, y_train,y_test = train_test_split(x,y, test_size =.2)

#create and train the support vector machince regressor
svr_rbf = SVR(kernel='rbf',C=1e3,gamma=.1)
svr_rbf.fit(x_train,y_train)

#testing model
svm_confidence = svr_rbf.score(x_test,y_test)
print(svm_confidence)

#create and train the linear regression model
lr =LinearRegression()
lr.fit(x_train,y_train)

#test the model
lr_confidence = lr.score(x_test,y_test)
print(lr_confidence)

#set forecast = to the last 1 row
x_forecast =np.array(dfc.drop(['Prediction'],1))[-forecast_out:]

print(x_forecast)

#print the predictions for the next n days
lr_prediction = lr.predict(x_forecast)
print(lr_prediction)

plt.figure(figsize=(16,8))

dfc.columns = ['dt','y']
dfc['dt'] = pd.to_datetime(dfc['dt'])


plt.title('Curve Prediction')
plt.xlabel('Date')
plt.ylabel('Cases')

plt.plot(dfc)
plt.plot(lr_prediction)
plt.legend()
plt.show()

