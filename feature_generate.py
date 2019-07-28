# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 21:03:37 2019

@author: Mariya Korlyakova
"""

"""
Ax^2+Bxy+Cy^2+Dx+Ey+F=0
Ellipse if B2-4AC<0.
Parabola if B2-4AC=0.
Hyperbola if B2-4AC>0

"""

# Data Generator
import numpy as np
import matplotlib.pyplot as plt

# генератор ответов: выход 1 - регрессор, выход 2 - классификатор если ошибка или вcе входы 0 - вернем 2, иначе -1, 0, 1, 
def ConicSectionType(x):
    mask_x = x!=0
    # если что-то пойдет не так, или все  входs = 0
    y = 2
    if np.sum(mask_x)>0:
      y_if = x[1] ** 2 - x[0] * x[2] 
 # y_if for regression
      y = -1 #for classification
      if y_if == 0:
           y = 0
      if y_if>0:
           y = 1           
       
    return y_if, y
# Строим дата сет
def setY(x):
    y = np.zeros((x.shape[0],2))
    for i in range(x.shape[0]):
        y[i,0], y[i,1] = ConicSectionType(x[i,:])
    return y
# DataSet generation for [-CoefLimit, CoefLimit] диапазоны на число примеров Numb_of_Train- для тренировки, Numb_of_Test - тестирования 
def DataGen(Numb_of_Train, Numb_of_Test, CoefLimit):
    # bild input for DataSet
    xTrain = (np.random.random((Numb_of_Train*2,3)) - 0.5) * 2 * CoefLimit
    xTest = (np.random.random((Numb_of_Test*2,3)) - 0.5) * 2 * CoefLimit

    
    # bild 0 output b = +\- (a*c*4)^2 
    sign_ac =np.random.randint(0,2,size =(Numb_of_Train,1), dtype = 'int') * 2 - 1
    sign_ac_test = np.random.randint(0,2,size = (Numb_of_Test,1), dtype = 'int') * 2 - 1
    #print(sign_ac[:10],'\n')
    
    xTraina0 = (np.linspace(0, CoefLimit / 50,Numb_of_Train)).reshape((Numb_of_Train,1))   * sign_ac
    xTesta0 = (np.linspace(0, CoefLimit /50 ,Numb_of_Test)).reshape((Numb_of_Test,1)) * sign_ac_test    
    xTrainc0 = (np.linspace( CoefLimit /50, 0,Numb_of_Train)).reshape((Numb_of_Train,1))  * sign_ac
    xTestc0 = (np.linspace( CoefLimit /50, 0 ,Numb_of_Test)).reshape((Numb_of_Test,1)) * sign_ac_test
    #print('**',xTraina0.shape)
    
    #plt.plot(xTrainc0)
    #plt.show()
    
    n = Numb_of_Train//2
    sign_b_train=np.ones((Numb_of_Train,1))
    sign_b_train[:n] = sign_b_train[:n]*(-1)
    n = Numb_of_Test//2
    sign_b_test=np.ones((Numb_of_Test,1))
    sign_b_test[:n] = sign_b_test[:n]*(-1)
    
    xTrainb0 = (sign_b_train * xTraina0 * xTrainc0 * 4) 
    xTestb0 = (sign_b_test * xTesta0 * xTestc0 * 4) 
    
    #print(n,xTestb0[240:260])
    
    xTrain0=np.hstack((xTraina0,xTrainb0,xTrainc0))
    xTest0=np.hstack((xTesta0,xTestb0,xTestc0))


    
     # bild output for DataSet
    yTrain = setY( xTrain )
    yTest = setY( xTest )
    

    
    yTrain0 = np.zeros((Numb_of_Train,2))
    yTest0 = np.zeros((Numb_of_Test,2))
    
    
    
    xTrain_ = np.vstack((xTrain,xTrain0))
    xTest_ = np.vstack((xTest,xTest0)) 
    
    yTrain_ = np.vstack((yTrain,yTrain0))
    yTest_ = np.vstack((yTest,yTest0))
    
    plt.plot(xTrain_[:,2],yTrain_[:,0],'.r')
    
    plt.show()
    
    
    
    return xTrain_ , yTrain_ , xTest_ , yTest_


xTrain , yTrain , xTest , yTest = DataGen(5000, 500, 1000)

print(xTrain.shape,yTrain.shape)
TrainData = np.hstack((xTrain , yTrain))
TestData = np.hstack((xTest , yTest))

# test Generation
print(yTrain[ yTrain[:,1] == 2 ] )
print(xTrain[0,:],yTrain[0])

np.savetxt("train.csv", TrainData , delimiter=",")
np.savetxt("test.csv", TestData , delimiter=",")

xData = xTrain
yData = yTrain

print(xTrain.min(axis=0))
plt.subplot(321)
plt.plot(xData[:,0], yData[:,0], '.r')

plt.subplot(322)
plt.plot(xData[:,1], yData[:,0], '.g')


plt.subplot(323)
plt.plot(xData[:,2], yData[:,0],'.b')

plt.subplot(324)
plt.plot(xData[:,0], yData[:,1], '.r')

plt.subplot(325)
plt.plot(xData[:,1], yData[:,1], '.g')


plt.subplot(326)
plt.plot(xData[:,2], yData[:,1],'.b')



plt.show()

xData = xTest
yData = yTest

print(xTrain.min(axis=0))
plt.subplot(321)
plt.plot(xData[:,0], yData[:,0], '.r')

plt.subplot(322)
plt.plot(xData[:,1], yData[:,0], '.g')


plt.subplot(323)
plt.plot(xData[:,2], yData[:,0],'.b')

plt.subplot(324)
plt.plot(xData[:,0], yData[:,1], '.r')

plt.subplot(325)
plt.plot(xData[:,1], yData[:,1], '.g')


plt.subplot(326)
plt.plot(xData[:,2], yData[:,1],'.b')



plt.show()
