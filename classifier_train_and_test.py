import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


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



TrainData = np.loadtxt("train.csv",  delimiter=",")
TestData = np.loadtxt("test.csv",  delimiter=",")

# f:eature 

xData = TrainData[:,:-2]
yData = TrainData[:,-2:]

xDataTest = TestData[:,:-2]
yDataTest = TestData[:,-2:]

quadratic = PolynomialFeatures(degree=2)

X_Data_quad = quadratic.fit_transform(xData)


model_best = RandomForestRegressor( n_estimators=50, max_features ='sqrt', random_state=1) # случайный лес

model_best.fit(X_Data_quad, yData[:,0]) 

X_DataTest_quad = quadratic.fit_transform(xDataTest)

y_Train_pred = model_best.predict(X_Data_quad)

#zeroo level
porog0 = np.std(y_Train_pred[:-1000]) ** 0.5

y_Test_class = solver_(model_best, porog0, X_DataTest_quad)
y_Train_class = solver_(model_best, porog0, X_Data_quad)

y_Test_class=y_Test_class.astype(dtype='int8')
yD=np.array(yDataTest[:,1]).astype(dtype='int8')

error_test = np.sum( yD != y_Test_class) / xDataTest.shape[0] 
error_train = np.sum( yData[:,1] != y_Train_class) / xData.shape[0]
#print('доля ошибок', np.sum( yDataTest[:,1] != y_Test_class) / xDataTest.shape[0] )
class_name = ["эллипс","параб.","гипер."]
f = open('testres.txt', 'w')
f.write('доля ошибок для обучения в % : '+str(error_train*100) + '\n')
f.write('доля ошибок для test в %: '+str(error_test*100) + '\n')

f.write('\nпо итогам анализа использую модель : RandomForestRegressor( n_estimators=50, random_state=1) # случайный лес')

f.write('\nгипер-параметры настроены без особого упорства, главное был выбор схемы обработки результатов регрессора в классификатор, обработка входов - полиномизация (2 порядок), как результат анализа нескольких моделей предобработки ')
f.write('\n             A            B                     C                 target_rez                my_rez')
for i  in range(y_Test_class.shape[0]):
    f.write('\n A:'  + str(xDataTest[i,0]) + '  B:' + str(xDataTest[i,1]) + '  C:' + str(xDataTest[i,2]) + '  y_true:'+class_name[yD[i]+1] +  '  y_pred:' + class_name[y_Test_class[i] + 1])
    
f.close()

#plt.plot(y_Test_pred,'.r')

#plt.plot(yDataTest[:,0],'.g')
#plt.show


with open('my_best_model.pkl', 'wb') as fid:
    pickle.dump(model_best, fid)  