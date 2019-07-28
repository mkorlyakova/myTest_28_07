import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

#solver
#solver
def solver_(model_best = None, porog0 = 0 , X_quad = []):
  mask1=[]
  y_pred = model_best.predict(X_quad)
  
  y_class = (-1)*np.ones(X_quad.shape[0])
  mask1 = np.where(np.abs(y_pred) < porog0)
  y_class[mask1[0]] = 0
  mask1 = np.where(y_pred >= porog0)
  y_class[mask1[0]] = 1
  return  y_class

# load it again
with open('my_best_model.pkl', 'rb') as fid:
    model_best = pickle.load(fid)

TrainData = np.loadtxt("train.csv",  delimiter=",")
TestData = np.loadtxt("test.csv",  delimiter=",")

# error  find
mask2 = np.array(TrainData[:,TrainData.shape[-1]-1] == 2 )
mask_nan = np.isnan(TrainData[:,TrainData.shape[-1]-1])
mask_inf = np.isinf(TrainData[:,TrainData.shape[-1]-1])

mask =  (mask2 + mask_nan + mask_inf) == 0


TrainData = TrainData[mask,:]
# f:eature 

xData = TrainData[:,:-2]
yData = TrainData[:,-2:]

xDataTest = TestData[:,:-2]
yDataTest = TestData[:,-2:]

quadratic = PolynomialFeatures(degree=2)
X_Data_quad = quadratic.fit_transform(xData)


#model_best = RandomForestRegressor( n_estimators=50, max_features ='sqrt', random_state=1) # случайный лес

X_DataTest_quad = quadratic.fit_transform(xDataTest)
X_Data_quad = quadratic.fit_transform(xData)
#classification level
y_Train_pred = model_best.predict(X_Data_quad)
porog0 = np.std(y_Train_pred[:-1000]) ** 0.5

y_Test_class = solver_(model_best, porog0, X_DataTest_quad)

plt.plot(yDataTest[:,1],'.r',label='true')

plt.plot(y_Test_class,'.g',label='test')
plt.xlabel('No sample')
plt.ylabel('y_classificator')
plt.legend()
plt.title('my classifier')
plt.show

print('доля ошибок %', np.sum( yDataTest[:,1] != y_Test_class) / xDataTest.shape[0] *100 )