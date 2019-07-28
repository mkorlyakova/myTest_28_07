# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 21:36:14 2019

@author: 111
"""
#проверяем работает ли как регрессия
#regressor
#Regression
import numpy as np
import pandas as pd
#from sklearn.linear_model import SVC 
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Embedding, Flatten, Activation
from tensorflow.python.keras.optimizers import Adam, RMSprop

from tensorflow.keras import utils

def model_net(neuron = 0 ,n_input = (3),n_out = 1, loss = "categorical_crossentropy", metric = "accuracy", optimizer = "adam",dropout = 0,activation_out = "softmax"):
    model = Sequential()
    model.add(Dense(n_input, input_dim=n_input, activation="relu"))
    if neuron>0 :
      if dropout > 0: model.add(Dropout(rate=dropout))
      model.add(Dense(neuron, activation="relu")) 
    if dropout > 0: model.add(Dropout(rate=0.005*dropout))
    model.add(Dense(n_out, activation=activation_out))
    model.compile(loss = loss, optimizer = optimizer , metrics = [metric])#"categorical_crossentropy",
    return model





TrainData = np.loadtxt("train.csv",  delimiter=",")

# error  find
mask2 = np.array(TrainData[:,TrainData.shape[-1]-1] == 2 )
mask_nan = np.isnan(TrainData[:,TrainData.shape[-1]-1])
mask_inf = np.isinf(TrainData[:,TrainData.shape[-1]-1])

mask =  (mask2 + mask_nan + mask_inf) == 0


TrainData = TrainData[mask,:]

# f:eature 

xData = TrainData[:,:-2]
yData = TrainData[:,-2:]


# можно брать класификатор, но признаков мало и они не показывают какой -то значимой зависимости
# построю регрессор
# считаем, что знаю только TrainData, данных много - сделаем сплит
sc = StandardScaler()
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(xData)
X_cubic = cubic.fit_transform(xData)
X_St=sc.fit_transform(xData)
X_quad_St = quadratic.fit_transform(X_St)
X_cubic_St = cubic.fit_transform(X_St)


X_list =[xData, X_St,X_quad_St, X_cubic_St]#X_quad, X_cubic,

models = [LinearRegression(), # метод наименьших квадратов
          RandomForestRegressor(n_estimators=50, max_features ='sqrt', random_state=1), # случайный лес
          KNeighborsRegressor(n_neighbors=6), # метод ближайших соседей
          SVR(kernel='rbf'), # метод опорных векторов с rbf ядром
          GradientBoostingRegressor(n_estimators=25, random_state=1),# немного ансамбля
          Ridge(random_state=1) #ну и Ридж
          
          ]


 
TestModels = pd.DataFrame()
tmp = {}
for j in range(len(X_list)):
  X_train, X_test, y_train, y_test = train_test_split(X_list[j], yData, test_size=0.25, random_state=42)
  
  
  models_neuron = [10 , 100 , 200 ]

#для каждой модели из списка model_neuron
  for model_n in models_neuron:
    print(model_n,X_train[0].shape)
    #получаем имя модели
    model = model_net(neuron = model_n ,n_input = (X_train[0].shape[0]),n_out = 1, loss = "mse", metric = "mse", optimizer = "adam",dropout = 0.1,activation_out = "linear")
    
    tmp['Model'] = 'model_net'  + str(model_n)
    #для каждого столбцам результирующего набора
    i = 0
    #обучаем модель
    model.fit(X_train, y_train[:,i], batch_size=100, epochs=30, verbose=0) 
    #вычисляем оценку качества
    tmp['R2_Y%s'%str(i+1)] = mean_absolute_error(y_test[:,i], model.predict(X_test))
    tmp['out No'] = i
    tmp['input No'] = j

    print('input No',j,'out No', i,'оценка',tmp['R2_Y%s'%str(i+1)])
    #записываем данные и итоговый DataFrame
    TestModels = TestModels.append([tmp])
  #для каждой модели из списка clasifier
  for model in models:
    
    #получаем имя модели
    m = str(model)
    print(model)
    tmp['Model'] = m[:m.index('(')]    
    #для каждого столбцам результирующего набора
    i = 0
    #обучаем модель
    model.fit(X_train, y_train[:,i]) 
    #вычисляем оценку качества
    tmp['R2_Y%s'%str(i+1)] = mean_absolute_error(y_test[:,i], model.predict(X_test))
    tmp['out No'] = i
    tmp['input No'] = j

    print('input No',j,'out No', i,'оценка',tmp['R2_Y%s'%str(i+1)])
    #записываем данные и итоговый DataFrame
    TestModels = TestModels.append([tmp])


print(TestModels.groupby(['Model'])['R2_Y1'].describe())

# Best model RandomForestRegressor
# input No 1 out No 0 оценка 1.0  (входные данные обработали в полиномиальную модель (2-порядок), выход по регресии)

#проверим классификацию
#Classificator


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


TrainData = np.loadtxt("train.csv",  delimiter=",")

# error  find
mask2 = np.array(TrainData[:,TrainData.shape[-1]-1] == 2 )
mask_nan = np.isnan(TrainData[:,TrainData.shape[-1]-1])
mask_inf = np.isinf(TrainData[:,TrainData.shape[-1]-1])

mask =  (mask2 + mask_nan + mask_inf) == 0


TrainData = TrainData[mask,:]
# f:eature 

xData = TrainData[:,:-2]
yData = TrainData[:,-2:]


sc = StandardScaler()
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(xData)
X_cubic = cubic.fit_transform(xData)
X_St=sc.fit_transform(xData)
X_quad_St = quadratic.fit_transform(X_St)
X_cubic_St = cubic.fit_transform(X_St)


X_list =[xData, X_St,X_quad_St, X_cubic_St]#X_quad, X_cubic,



models = [  KNeighborsClassifier(6),
            SVC(kernel="rbf", C=0.025),
            #GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1),
            AdaBoostClassifier(),
            GaussianNB()
          ]
 
TestModelsClass = pd.DataFrame()
tmp = {}
for j in range(len(X_list)):
  X_train, X_test, y_train, y_test = train_test_split(X_list[j], yData, test_size=0.25, random_state=42)
  
  y_train_net = utils.to_categorical(y_train[:, 1],3)
  y_test_net = utils.to_categorical(y_test[:, 1],3)
  models_neuron = [10 , 100 , 200 ]

#для каждой модели из списка clasifier
  for model_n in models_neuron:
    print(model_n,X_train[0].shape)
    #получаем имя модели
    model = model_net(neuron = model_n ,n_input = (X_train[0].shape[0]),n_out = 3,dropout = 0.1)
    
    tmp['Model'] = 'model_net'  + str(model_n)
    #для каждого столбцам результирующего набора
    #обучаем модель
    model.fit(X_train, y_train_net, batch_size=100, epochs=30, verbose=0) 
    #вычисляем оценку качества
    tmp['R2_Y%s'%str(i+1)] = mean_absolute_error(y_test_net, model.predict(X_test))
    tmp['out No'] = i
    tmp['input No'] = j

    print('input No',j,'out No', i,'оценка',tmp['R2_Y%s'%str(i+1)])
    #записываем данные и итоговый DataFrame
    TestModelsClass = TestModelsClass.append([tmp])

  #для каждой модели из списка
  for model in models:
    
    #получаем имя модели
    m = str(model)
    print(model)
    tmp['Model'] = m[:m.index('(')]    
    #для каждого столбцам результирующего набора
    i = 1
    #обучаем модель
    model.fit(X_train, y_train[:,i]) 
    #вычисляем оценку качества
    tmp['R2_Y%s'%str(i+1)] = r2_score(y_test[:,i], model.predict(X_test))
    tmp['out No'] = i
    tmp['input No'] = j
    print('out No', i,'оценка',tmp['R2_Y%s'%str(i+1)])
    #записываем данные и итоговый DataFrame
    TestModelsClass = TestModelsClass.append([tmp])
    
# лучший классификатор нелинейный перептрон (100 нейронов) MLPClassifier	на прямой модели XData	с оценкой 0.90    (можно погонять и получить лучше, но пока регрессия красивее)

print(TestModelsClass.groupby(['Model'])['R2_Y2'].describe())


#буду брать регрессию
