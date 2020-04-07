#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#from numpy import linalg as LA 

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target

    
a = [[1,2,3],[4,5,6],[7,8,9]]
b = np.array([1,2,3])
b = np.transpose(b)
#a = np.sum(a, axis = 0)
#a = np.sum(a, axis = 0)

#trainData, trainTarget, validData, validTarget, testData, testTarget = loadData()
x_train, x_valid, x_test, y_train, y_valid, y_test = loadData()
y_train, y_valid, y_test = convertOneHot(y_train, y_valid, y_test)

#trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)

xtr_numEntries = x_train.shape[0]
xtr_numFeatures1 = x_train.shape[1] 
xtr_numFeatures2 = x_train.shape[2]  
x_train = x_train.flatten()
x_train = x_train.reshape(xtr_numEntries, xtr_numFeatures1*xtr_numFeatures2)
x_train = np.transpose(x_train) 

xva_numEntries = x_valid.shape[0]
xva_numFeatures1 = x_valid.shape[1] 
xva_numFeatures2 = x_valid.shape[2]  
x_valid = x_valid.flatten()
x_valid = x_valid.reshape(xva_numEntries, xva_numFeatures1*xva_numFeatures2)
x_valid = np.transpose(x_valid) 

xte_numEntries = x_test.shape[0]
xte_numFeatures1 = x_test.shape[1] 
xte_numFeatures2 = x_test.shape[2]  
x_test = x_test.flatten()
x_test = x_test.reshape(xte_numEntries, xte_numFeatures1*xte_numFeatures2)
x_test = np.transpose(x_test) 


w_h = np.random.normal(0, 2/1010, (784, 1000))
w_o = np.random.normal(0, 2/1010, (1000, 10))
b_h = np.random.normal(0, 2/1010, (1, 1000))
b_o = np.random.normal(0, 2/1010, (1, 10))

'''The initializations below are for the training data, 
for the other data sets, matrices and dimensions will have to be changed'''


def relu(x): 
    x = np.maximum(x, 0)
    
    return x  
    

def softmax(x):
    max_elem =  np.max(x)
    x = np.subtract(x, max_elem)
    
    exp_elem = np.exp(x)  
    exp_sum_cols = np.sum(exp_elem, axis = 0)[:,None]
    exp_sum_cols = np.transpose(exp_sum_cols)
        
    sm = np.divide(exp_elem, exp_sum_cols)  
    return sm  


def computeLayer(X, W, b):
    
    w_t = np.transpose(W)
    pred = np.matmul(w_t, X) 
    #pred +=  np.transpose(b) 
    return pred


def CE(target, prediction):
    
    prediction = np.log(prediction)
    ce = np.multiply(target, prediction)
    ce = (-1/10000)*(np.sum(ce))
    
    return ce
       
#p = computeLayer(trainData, w_h, b_h) 
#p = relu(p)
#p = computeLayer(p, w_o, b_o) 
#p = softmax(p)   
    
def gradCE(target, prediction):
    
    one = np.ones(10000)  
    P_Y = prediction - target
    
    lay1 = computeLayer(x_train, w_h, b_h)
    H = relu(lay1)
            
    grad_w = np.matmul(H, P_Y) 
    grad_b = np.matmul(one, P_Y)
    
    return grad_b, grad_w
'''   
layer_1 = computeLayer(x_train, w_h, b_h) 
H = relu(layer_1)
O = computeLayer(w_o, H, b_o)   
P = softmax(O)
'''
  
def grad_descent(data, target, alpha = 10**(-5) , gamma = 0.99):  
    
    w_h = np.random.normal(0, 2/1010, (784, 1000))
    w_o = np.random.normal(0, 2/1010, (1000,10))
    b_h = np.random.normal(0, 2/1010, (1, 1000))
    b_o = np.random.normal(0, 2/1010, (1, 10))
    
    v_wh = np.full((784,1000), 10**(-5))
    v_wo = np.full((1000,10), 10**(-5))
    v_bo = np.full((1,10), 10**(-5))
    v_bh = np.full((1,1000), 10**(-5))

    
    layer_1 = computeLayer(data, w_h, b_h) 
    H = relu(layer_1)
    O = computeLayer(w_o, H, b_o)   
    P = softmax(O)
    
    
    
    P_Y = (P - target)
    one_grad_term = np.ones(10000)
    grad_term_1 = np.matmul(P_Y, np.transpose(w_o))
    
    grad_wh = np.matmul(data, grad_term_1) 
    grad_bh = np.matmul(one_grad_term, grad_term_1) 
    
    grad_wo = gradCE(target, P)[1]
    grad_bo = gradCE(target, P)[0]  

    iteration = []
    loss = []  
    accuracy = [] 
       
    for i in range(200):  
        
        v_wh = (gamma)*v_wh + (alpha)*grad_wh 
        w_h = w_h - v_wh 
        
        v_wo = (gamma)*v_wo + (alpha)*grad_wo 
        w_o = w_o - v_wo 
        
        v_bh = (gamma)*v_bh + (alpha)*grad_bh 
        b_h = b_h - v_bh  
        
        v_bo = (gamma)*v_bo + (alpha)*grad_bo  
        b_o = b_o - v_bo  
        
        layer_1 = computeLayer(data, w_h, b_h) 
        H = relu(layer_1)
        O = np.matmul(np.transpose(w_o), H)#computeLayer(np.transpose(w_o), H, b_o)   
        O += np.transpose(b_o)  
        P = softmax(O)
        
    
        P_Y = (P - np.transpose(target))
        grad_term_1 = np.matmul(np.transpose(P_Y), np.transpose(w_o))
    
        grad_wh = np.matmul(data, grad_term_1) 
        grad_bh = np.matmul(one_grad_term, grad_term_1) 
    
        grad_wo = gradCE(target, np.transpose(P))[1]
        grad_bo = gradCE(target, np.transpose(P))[0]

        
        final_pred = np.argmax(P, axis = 0)
        label = np.argmax(np.transpose(target), axis = 0)
        acc = np.mean(final_pred == label) 
        
        accuracy.append(acc)
        CElossvalue = CE(np.transpose(target), P)
        loss.append(CElossvalue)
        iteration.append(i)
        
       
     
    loss =  np.reshape(loss, 200)
    iteration = np.reshape(iteration, 200)
    accuracy = np.reshape(accuracy, 200)
     
    '''
    ##
    plt.title("Loss vs Epoch") 
    plt.xlabel("Epoch") 
    plt.ylabel("Loss") 
    plt.plot(iteration,loss) 
    plt.show() 
    ## 
    '''
    
    #use this 
    plt.title("Accuracy vs Epoch") 
    plt.xlabel("Epoch") 
    plt.ylabel("Accuracy") 
    plt.plot(iteration, accuracy) 
    plt.show()   
    
    
        
    return #w_h, w_o, b_h, b_o
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

