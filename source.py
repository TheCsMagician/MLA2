# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:41:39 2018

@author: zainu
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc
import pickle 
import sklearn.linear_model as lin
from sklearn.neighbors import KNeighborsClassifier
from numpy.random import randn


#Question 1a:
def genData(mu0,mu1,Sigma0,Sigma1,N): 
    t0 = np.full((N,1),0)
    t1 = np.full((N,1),1)
    t = np.append(t0,t1,axis = 0)
    x0 = np.random.multivariate_normal(mu0,Sigma0,N)
    x1 = np.random.multivariate_normal(mu1,Sigma1,N)
    x = np.append(x0,x1,axis=0)
    suff_x, suff_t = shuffle(x,t)
    return suff_x,suff_t
#Question 1b:
    
mu0 = [0,-1]
mu1 = [-1,1]
Sigma0 = np.matrix('2.0 0.5; 0.5 1.0')
Sigma1 = np.matrix('1.0 -1.0; -1.0 2.0')
x, t=  genData(mu0,mu1,Sigma0,Sigma1, 10000)

#Question 1c:
def plotmaker(x,t,s):
    plt.title(s)
    for i in range(len(t)):
        if(t[i] == 0):
            plt.scatter(x[i,0],x[i,1],s=2,c='r')
        else: 
            plt.scatter(x[i,0],x[i,1],s=2,c='b')
    plt.xlim(-5,6)
    plt.ylim(-5,6)
    return plt

plotmaker(x,t,"Question 1(c): sample cluster data (10,000 points per cluster)").show()

#Question 2a:
x, t=  genData(mu0,mu1,Sigma0,Sigma1, 1000)
#Question 2b:
clf = LogisticRegression().fit(x,t)
w0 = clf.intercept_
w1 = clf.coef_
m = clf.score(x,t)
print "w0 = ", w0 , "w1 = ", w1, "mean accuracy= ", m
#Question 2c:
p = plotmaker(x,t,"Question 2(c): training data and decision boundary")
x = np.linspace(-5,6)
y = -(w1[0][0] * x)/w1[0][1] + w0[0] / w1[0][1]
p.plot(x,y,c= 'black')
p.show()
#Question 2d:
x, t=  genData(mu0,mu1,Sigma0,Sigma1, 1000)
p = plotmaker(x,t,"Question 2(d): decision boundaries for seven thresholds")
for g in range(-3,4):
    y = -(w1[0][0] * x + g)/w1[0][1] + w0[0] / w1[0][1]
    if g ==0:
        p.plot(x,y,c='k')
    elif g <0:
        p.plot(x,y,c='b')
    else:
        p.plot(x,y,c='r')
p.show()


#Question 2g:
x1, t1=  genData(mu0,mu1,Sigma0,Sigma1, 10000)
#Question 2h:
clf = LogisticRegression().fit(x1,t1)
w0 = clf.intercept_
w1 = clf.coef_
m = clf.score(x1,t1)
x = np.linspace(-5,6,20000)
y = -(w1[0][0] * x1[:,0] - 1)/w1[0][1] + w0[0] / w1[0][1]
num_positive = 0.0
num_negative = 0.0
for i in range(len(t1)):
    if(x1[i,1] > y[i]):
        num_positive = num_positive +1
    else:
        num_negative = num_negative +1
print "class0 = red"
print "class1 = blue"
print "The number of predicted positives (i.e., points predicted to be in class 1) = ", num_positive
print "The number of predicted negatives (i.e., points predicted to be in class 0) =", num_negative

true_pos = 0.0
false_pos= 0.0
true_neg = 0.0
false_neg = 0.0

plt.title("Question 2(h): explanatory figure")
plt.xlim(-5,6)
plt.ylim(-5,6)
plt.scatter(x1[0,0],x1[0,1],s=2,c='blue',label = "true positive")
plt.scatter(x1[0,0],x1[0,1],s=2,c='green', label = "false positive")
plt.scatter(x1[0,0],x1[0,1],s=2,c='red' , label = "true negative")
plt.scatter(x1[0,0],x1[0,1],s=2,c='purple',label= "false negative")

for i in range(len(t1)):
    
    
    if(x1[i,1] > y[i]):
        if(t1[i] == 1):
            true_pos = true_pos +1
            plt.scatter(x1[i,0],x1[i,1],s=2,c='blue')
        else:
            false_pos = false_pos +1
            plt.scatter(x1[i,0],x1[i,1],s=2,c='green')
    else:
        if(t1[i] == 0):
            true_neg = true_neg+1
            plt.scatter(x1[i,0],x1[i,1],s=2,c='red')
        else:
            false_neg = false_neg +1
            plt.scatter(x1[i,0],x1[i,1],s=2,c='purple')
x = np.linspace(-5,6,20000)
y = -(w1[0][0] * x - 1)/w1[0][1] + w0[0] / w1[0][1]

plt.plot(x,y,c='black')
plt.show()
print "The number of true positives (i.e., predictions for class 1 that are correct) = ", true_pos
print "The number of false postives (i.e., predictions for class 1 that are incorrect) = ", false_pos
print "The number of true negatives (i.e., predictions for class 0 that are correct) = ", true_neg
print "The number of false negatives (i.e., predictions for class 0 that are incorrect) = ", false_neg
print "The recall = ", true_pos/(true_pos+false_neg) *100
print "The precision = ", true_pos/(true_pos+false_pos) *100

#Question 2f:

x1, t1=  genData(mu0,mu1,Sigma0,Sigma1, 1000)
clf = LogisticRegression().fit(x1,t1)
w0 = clf.intercept_
w1 = clf.coef_
m = clf.score(x1,t1)
pres = []
recall = []
for i in range(0,1):
    true_pos = 0.0
    false_pos= 0.0
    true_neg = 0.0
    false_neg = 0.0
    y = -(w1[0][0] * x1[:,0] - i)/w1[0][1] + w0[0] / w1[0][1]
    for i in range(len(t1)):
        if(x1[i,1] > y[i]):
            if(t1[i] == 1):
                true_pos = true_pos +1
            
            else:
                false_pos = false_pos +1
           
        else:
            if(t1[i] == 0):
                true_neg = true_neg+1
            
            else:
                false_neg = false_neg +1
    pres1 = true_pos/(true_pos+false_pos)
    recall1 = true_pos/(true_pos+false_neg)
    pres.append(pres1)
    recall.append(recall1)  
plt.plot(recall,pres,c='blue')
plt.xlabel("recall")
plt.ylabel("precision")
plt.title("Question 2(i): precision/recall curve")
plt.show()

#Question 2j:    
#Question 2k:
total = 0
for i in range(1,len(recall)):
    total = total + (recall[i-1] - recall[i]) * pres[i]
    
print "The area under precision/recall curve", total

#Question 3a:
with open("mnist.pickle","rb") as f:
    Xtrain,Ytrain,Xtest,Ytest = pickle.load(f)

Xtrain1 = shuffle(Xtrain)

fig=plt.figure(figsize=(6, 6))
for i in range(36):
    fig.add_subplot(6,6,i+1)
    plt.axis('off')
    plt.imshow(np.reshape(Xtrain[i],(28,28)),cmap='Greys',interpolation='nearest')
plt.title("Question 3(a): 36 random MNIST images.")
plt.show()

#Question 3b:
clf = lin.LogisticRegression(multi_class='multinomial',solver='lbfgs')
clf.fit(Xtrain,Ytrain)
mean_acc_trian_data = clf.score(Xtrain,Ytrain)
mean_acc_test_data  = clf.score(Xtest,Ytest)
print '\n Question 3(b):'
print "mean accuracy of training data ", mean_acc_trian_data*100 
print "mean accuracy of testing data ", mean_acc_test_data*100


#Question 3c:
kv= []
mean_test = []
best_k = 0
best_mean = 0
k_val = range(1,21)
for k in range(1,21):
   KNN = KNeighborsClassifier(algorithm= 'brute',n_neighbors = k).fit(Xtrain,Ytrain)
   mean_acc_test = KNN.score(Xtest,Ytest)
   mean_test.append(mean_acc_test*100)
   kv.append(k)
   if(mean_acc_test > best_mean):
       best_mean = mean_acc_test
       best_k = k
   
plt.plot(kv,mean_test)
plt.title("Figure 3c KNN test accuracy")
plt.show()
print "best k = ", best_k
print "best mean = ", best_mean


with open("mnist.pickle","rb") as f:
    Xtrain,Ytrain,Xtest,Ytest = pickle.load(f)

#Question 5a:
def softmax1(z):
    a = np.exp(z)
    b = np.sum(np.exp(z) ,axis = 0)
    return np.divide(a,b)
print "softmax1 (0,0) = ", softmax1((0,0))
print "softmax1 (1000,0) = ", softmax1((1000,0))
print "softmax1 (-1000,0) = ", softmax1((-1000,0))

#Question 5c:
def softmax2(z):
    y = np.subtract(z,np.max(z))
    n = np.exp(y)
    d = np.sum(n)
    l = np.exp(y) - np.log(np.sum(np.exp(y)))
    return np.divide(n,d), l
print "softmax2 (0,0) = ", softmax2((0,0))
print "softmax2 (1000,0) = ", softmax2((1000,0))
print "softmax2 (-1000,0) = ", softmax2((-1000,0))

print "Question 6: I don't know"
print "Question 7: I don't know"
