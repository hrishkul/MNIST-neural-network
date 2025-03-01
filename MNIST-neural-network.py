#import all libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as pp 

#LOADING AND SPLITTING DATA
data = pd.read_csv('mnist_train.csv') 

data = np.array(data) 
m, n=data.shape 
np.random.shuffle(data) 

data_dev=data[0:1000].T 
Y_dev=data_dev[0] 
X_dev=data_dev[1:n] 
X_dev=X_dev/255 

data_train=data[1000:m].T 
Y_train=data_train[0] 
X_train=data_train[1:n] 
X_train=X_train/255 
_,m_train = X_train.shape 


#INITIALIZING PARAMETERS
def params(): 
    W1=np.random.rand(10,784)-0.5
    b1=np.random.rand(10,1)-0.5
    W2=np.random.rand(10,10)-0.5
    b2=np.random.rand(10,1)-0.5
    return W1,b1,W2,b2


#FORWARD PROPAGATION STARTS
def ReLU(Z): 
    return np.maximum(Z,0)  

def softmax(Z): 
    A=np.exp(Z)/sum(np.exp(Z))
    return A 


def forward_prop(W1, b1, W2, b2, X): 
    Z1=W1.dot(X)+b1 
    A1=ReLU(Z1) 
    Z2=W2.dot(A1)+b2 
    A2=softmax(Z2) 

    return Z1, A1, Z2, A2

#BACK PROPAGATION STARTS 
def one_hot(Y):  
    one_hot_Y=np.zeros((Y.size,Y.max()+1)) 
    one_hot_Y[np.arange(Y.size),Y]=1 
    return one_hot_Y.T

def deriv_ReLU(Z): 
    return Z>0 

def back_prop(Z1, A1, Z2, A2, W2, X, Y): 
    m=Y.size 
    one_hot_Y=one_hot(Y) 
    dZ2=A2-one_hot_Y 
    dW2=1/m*dZ2.dot(A1.T) 
    db2=1/m*np.sum(dZ2)
    dZ1=W2.T.dot(dZ2)*deriv_ReLU(Z1) 
    dW1=1/m*dZ1.dot(X.T) 
    db1=1/m*np.sum(dZ1) 
    return dW1,db1,dW2,db2

#UPDATING PARAMETERS
def update_paras( W1, b1, W2, b2, alpha, dW1, db1, dW2, db2): 
    W1=W1-alpha*dW1 
    b1=b1-alpha*db1 
    W2=W2-alpha*dW2
    b2=b2-alpha*db2

    return W1,b1,W2,b2

#GRADIENT DESCENT OR TRAINING
def get_predictions(A2): 
    return np.argmax(A2, 0) 

def get_accuracy(predictions, Y): 
    print(predictions, Y)
    return np.sum(predictions==Y)/Y.size 

def gradient_descent(X, Y, iterations, alpha): 
    W1,b1,W2,b2=params() 
    for i in range(iterations): 
        Z1,A1,Z2,A2=forward_prop(W1,b1,W2,b2,X) 
        dW1,db1,dW2,db2=back_prop(Z1,A1,Z2,A2,W2,X,Y) 
        W1,b1,W2,b2=update_paras(W1,b1,W2,b2,alpha,dW1,db1,dW2,db2) 
        if (i%100==0): 
            print('iteraion - ',i)
            predictions=get_predictions(A2)
            print('accuracy - ', get_accuracy(predictions, Y))
        
    return W1,b1,W2,b2

W1,b1,W2,b2=gradient_descent(X_train, Y_train, 500, 0.10) 

#TESTING
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2) 
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_dev[:, index, None] 
    prediction = make_predictions(X_dev[:, index, None], W1, b1, W2, b2)
    label = Y_dev[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255 
    pp.gray() 
    pp.imshow(current_image, interpolation='nearest')
    pp.show()

test_prediction(0, W1 ,b1, W2, b2)
test_prediction(1, W1 ,b1, W2, b2)
test_prediction(2, W1 ,b1, W2, b2)
test_prediction(3, W1 ,b1, W2, b2)
test_prediction(4, W1 ,b1, W2, b2)
test_prediction(5, W1 ,b1, W2, b2)
test_prediction(6, W1 ,b1, W2, b2)
test_prediction(7, W1 ,b1, W2, b2)
test_prediction(8, W1 ,b1, W2, b2)
test_prediction(9, W1 ,b1, W2, b2)







