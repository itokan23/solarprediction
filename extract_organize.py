# -*- coding: utf-8 -*-
"""
Created on Sat Sep 02 17:58:25 2017

@author: Kan Ito
itokan@berkeley.edu
Solar Prediction from Kaggle
"""

#%% Librairies
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime as datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from math import sqrt
import matplotlib
#%% extract data, clean, convert to datetime,
data = pd.read_csv('SolarPrediction.csv',parse_dates=True,index_col=0)
print data.head()
print data[-10:]
""" Information/Notes
5 min time series
Columns: UNIXTime, Date, Time, Radiation [W/m^2], Temperature [F], 
Humidity [%], Barometric Pressure [Hg], Wind Direction [degrees], 
Wind Speed [mph], Sunrise/sunset [Hawaii Time]
Dates go from 9/29/2016 to 12/1/2016

"""
column_names = ['Date','Time','Radiation','Temperature', 'Pressure', 'Humidity', 'WindDir', 'Speed','TimeSunrise','TimeSunset']
data.columns = column_names
data['Date'] = pd.to_datetime(data['Date'])
data['TimeSunrise'] = pd.to_datetime(data['TimeSunrise'])
data['TimeSunset'] = pd.to_datetime(data['TimeSunset'])

print data.shape
print data.info()
#%% Visuals
plt.figure(1)
plt.scatter(data['Humidity'],data['Radiation'], s=10, color='r')
plt.title("Humidity v Radiation")
plt.ylabel("Radiation [W/m^2]")
plt.xlabel("Humidity [%]")
plt.show()
plt.figure(2)
plt.scatter(data['Temperature'],data['Radiation'],s=10,color='b')
plt.title("Temperature v Radiation")
plt.xlabel("Temperature [F]")
plt.ylabel("Radiation [W/m^2]")
plt.show()
plt.figure(3)
plt.scatter(data['Pressure'],data['Radiation'],s=10,color='m')
plt.title("Pressure v Radiation")
plt.xlabel("Pressure [Hg]")
plt.ylabel("Radiation [W/m^2]")
plt.show()
plt.figure(4)
plt.scatter(data['Speed'],data['Radiation'],s=10,color='g')
plt.title("Wind Speed v Radiation")
plt.xlabel("Wind Speed [mph]")
plt.ylabel("Radiation [W/m^2]")
plt.show()

#%% time series visuals
plt.figure(5)       
data['Radiation'].plot(color='r')
plt.figure(6)
data['Radiation'].ix[1475226323:1473827706].plot(color='r')
#%% Histogram
#data['Radiation'].hist(bins=30)
plt.figure(7)
nonzero_rad = data['Radiation'] > 50
nonzero_rad = data[nonzero_rad]
nonzero_rad['Radiation'].hist(color='r',bins=50)
#%% Look at correlation and covariance
print data.cov()
print data.corr()
print data['Radiation'].describe()
# strong coorelations to radiation: temp, weak: humidity, winddir
#%% Multiregression with vars: temp, humidity, windir
features = ['Temperature','Humidity','WindDir']
target = ['Radiation']
x = data[features]
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=324)
regressor = LinearRegression()
regressor.fit(x_train,y_train)
# Prediction
y_prediction = regressor.predict(x_test)
print y_prediction
print y_test.describe()
# Accuracy
RMSE = sqrt(mean_squared_error(y_test['Radiation'],y_prediction))
var_score = r2_score(y_test['Radiation'],y_prediction)
print "Accuracy RMSE: ", RMSE
print "Variance Score: ", var_score #coef of determination
print "Regression Coef (Temp, Hum, WindDir): ", regressor.coef_
print "Intercept: ",regressor.intercept_
#%% visualize results
plt.figure(8)
plt.scatter(x_test['Temperature'],y_prediction,s=5,color='b')
plt.scatter(x_test['Temperature'],y_test,s=5,color='r')
plt.title('Multiple Regression')
plt.xlabel('Temperature')
plt.ylabel('Radiation')
plt.figure(9)
plt.scatter(x_test['Humidity'],y_prediction,s=5,color='b')
plt.scatter(x_test['Humidity'],y_test,s=5,color='r')
plt.title('Multiple Regression')
plt.xlabel('Humidity')
plt.ylabel('Radiation')
plt.figure(10)
plt.scatter(x_test['WindDir'],y_prediction,s=5,color='b')
plt.scatter(x_test['WindDir'],y_test,s=5,color='r')
plt.title('Multiple Regression')
plt.xlabel('WindDir')
plt.ylabel('Radiation')
#%% Multiregression with discretized wind directions
"""The way WindDir is utiliized in Regression model makes little sense
 0deg is north, 90 is east, clockwise direction
 Location: Hawaii?"""
plt.figure(11)
data['WindDir'].hist(bins=25,color='r')
""" Very odd, data says heavy on southerly winds, some wind north and east, very little for west.
prevailing winds for Hawaii's latitudes says otherwise. Maybe its not Hawaii.
BUt sunrise and sunset is consistent with Hawaii
Strategy: divide into 4 sections: northeast, southeast,southwest, and northwest
"""
# create booleans for each directions
data['Northeast'] = data['WindDir'] < 91
data['Southeast'] = ((data['WindDir'] < 181) & (data['WindDir'] >= 91))
data['Southwest'] = ((data['WindDir'] < 271) & (data['WindDir'] >= 181))
data['Northwest'] = ((data['WindDir'] < 361) & (data['WindDir'] >= 271))
features = ['Temperature','Humidity','Northeast','Southeast','Southwest','Northwest']
target = ['Radiation']
x = data[features]
y = data[target]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state=323)
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_prediction = regressor.predict(x_test)
RMSE = sqrt(mean_squared_error(y_test['Radiation'],y_prediction))
var_score = r2_score(y_test['Radiation'],y_prediction)
print "Accuracy RMSE: ", RMSE
print "Variance Score: ", var_score #coef of determination
print "Regression Coef (Temp, Hum, NE,SE,SW,NW): ", regressor.coef_
print "Intercept: ",regressor.intercept_
# RESULT: slightly better, 0.03 COD improvement
#%% Visuals for Discretized Wind
plt.figure(12)
plt.scatter(x_test['Temperature'],y_prediction,s=5,color='b')
plt.scatter(x_test['Temperature'],y_test,s=5,color='r')
plt.title('Multiple Regression')
plt.xlabel('Temperature')
plt.ylabel('Radiation')
plt.figure(13)
plt.scatter(x_test['Humidity'],y_prediction,s=5,color='b')
plt.scatter(x_test['Humidity'],y_test,s=5,color='r')
plt.title('Multiple Regression')
plt.xlabel('Humidity')
plt.ylabel('Radiation')
#%% Try Decision Tree Regression better with larger depth
""" Can't look into the specifics of the tree but way good to be true R^2, must be very overfit"""
features = ['Temperature','Humidity','Northeast','Southeast','Southwest','Northwest']
target = ['Radiation']
x = data[features]
y = data[target]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state=322)
regre_1 = DecisionTreeRegressor(max_depth=2)
regre_2 = DecisionTreeRegressor(max_depth=5)
regre_1.fit(x,y)
regre_2.fit(x,y)
y_prediction1 = regre_1.predict(x_test)

y_prediction2 = regre_2.predict(x_test)
print "Decision Tree Regressor Depth=2"
print "Variance Score: ", regre_1.score(x_test,y_test) #coef of determination
print 
print "Decision Tree Regressor Depth=5"
print "Variance Score: ", regre_2.score(x_test,y_test) # coef  of D
print
plt.figure(14)
plt.scatter(x_test['Temperature'],y_prediction2,s=5,color='b', label = 'Model')
plt.scatter(x_test['Temperature'],y_test,s=5,color='r',label = 'Actual')
plt.legend(loc=4)
plt.title('Decision Tree Regression')
plt.xlabel('Temperature')
plt.ylabel('Radiation')
print regre_2.get_params(deep=True)
print regre_2.decision_path(x_train,check_input=True)

#%% 
""" Suggestions: ridge regression (regularization), Lasso, 
Polynomial Regression, also do something with sunset and sunrise, easy
also try to configure time
make copy of another csv and read with time index
"""
#%%
plt.figure(15)       
plt.scatter(data['Time'],data['Radiation'], color='r',s=5)
#%%
import matplotlib.dates as mdates
fig = plt.figure(15)
ax = fig.add_subplot(111)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y %H:%M'))
ax.plot_date(data['Time'][2000:3000],data['Radiation'][2000:3000],marker='o',color='orange')
ax.set_title('Radiation (Daily)')
ax.set_ylabel('Radiation [W/m^2]')
ax.set_xlabel('Time of day')
ax.grid(True)
plt.show()
#%% Toordinal

toordinal = data['Time'][1:2]
fck = toordinal.values[0]
shit = fck.toorindal()
#%% python 3.3 has pd.timestamp()
fck = data['Time']
shit = fck.Timestamp(data['Time'])
#%%
features = ['Temperature','Humidity','WindDir','Time']
target = ['Radiation']
x = data[features]
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=324)
regressor = LinearRegression()
regressor.fit(x_train,y_train)
# Prediction
y_prediction = regressor.predict(x_test)
print y_prediction
print y_test.describe()
# Accuracy
RMSE = sqrt(mean_squared_error(y_test['Radiation'],y_prediction))
var_score = r2_score(y_test['Radiation'],y_prediction)
print "Accuracy RMSE: ", RMSE
print "Variance Score: ", var_score #coef of determination
print "Regression Coef (Temp, Hum, WindDir): ", regressor.coef_
print "Intercept: ",regressor.intercept_
