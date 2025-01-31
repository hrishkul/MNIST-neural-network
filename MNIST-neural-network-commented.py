#import all libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as pp 

#load csv file in dataframe
data = pd.read_csv('mnist_train.csv') 

#LOADING AND SPLITTING DATA
data = np.array(data) #convert dataframe into numpy array
m, n=data.shape #identify rows and columns
np.random.shuffle(data) #shuffle all data

data_dev=data[0:1000].T #split the first 1000 rows as testing data and transpose it
Y_dev=data_dev[0] #split the first row as it is the label row(digits)
X_dev=data_dev[1:n] #split the rest as pixel data of the image
X_dev=X_dev/255 #scale the pixels to a range of 0 to 1


data_train=data[1000:m].T #split the remaining rows as testing data and transpose it
Y_train=data_train[0] #split the first row as it is the label row(digits)
X_train=data_train[1:n] #split the rest as pixel data of the image
X_train=X_train/255 #scale the pixels to a range of 0 to 1
_,m_train = X_train.shape #extract the number of examples in the training examples for gradients and biases


#INITIALIZING PARAMETERS
def params(): #function to initialise all parameters
    W1=np.random.rand(10,784)-0.5#assigns 0 or 1 randomly to all the input nodes same for all the statements in this function
    b1=np.random.rand(10,1)-0.5
    W2=np.random.rand(10,10)-0.5
    b2=np.random.rand(10,1)-0.5
    return W1,b1,W2,b2


#FORWARD PROPAGATION STARTS
def ReLU(Z): #(Activation Function)
    return np.maximum(Z,0)  #goes through each element and returns Z if the value is greater than 0 and 0 if the value is lesser than or equal to zero

def softmax(Z): #(Activation Function)
    A=np.exp(Z)/sum(np.exp(Z))
    return A #returns e to the power of each element divided by the sum of e to the power of each element


def forward_prop(W1, b1, W2, b2, X): #forward propagation function
    Z1=W1.dot(X)+b1 #Z1 is a hidden layer which is the dot product of the weight 1 and input matrices plus the bias 1
    A1=ReLU(Z1) #A1 is the hidden layer Z1 passed through the activation function ReLU
    Z2=W2.dot(A1)+b2 #Z2 is a hidden layer which is the dot product of the weight 2 and A1 plus the bias 2
    A2=softmax(Z2) #A2 is the hidden layer Z2 passed through the activation function softmax

    return Z1, A1, Z2, A2

#BACK PROPAGATION STARTS 
def one_hot(Y):  #function to implement one hot encoding to convert labels into binary vectors
    one_hot_Y=np.zeros((Y.size,Y.max()+1)) #creates a matrix filled with zeros with the shape (no. of elements in Y(label) array, number of categories(classes) plus one to handle all possible labels)
    one_hot_Y[np.arange(Y.size),Y]=1 #generates and array from 0 to Y.size-1 to select each row in the one_hot_Y matrix and sets the correct column corresponding to Y in each row to 1
    return one_hot_Y.T

def deriv_ReLU(Z): #function to get the derivative of the ReLU function
    return Z>0 #goes through each element and returns 1 if Z>0 and 0 if Z<0

def back_prop(Z1, A1, Z2, A2, W2, X, Y): #backward propagation function
    m=Y.size #initialises m as the total number of elements in Y
    one_hot_Y=one_hot(Y) #gets the one hot encoding of Y
    dZ2=A2-one_hot_Y #error of the second layer being calculate by subtracting the one hot value of the label from the predictions
    dW2=1/m*dZ2.dot(A1.T) #error of the weight of the second layer being calculated by multiplying the inverse of m by the dot product of the error of the second layer and the transpose of the hidden layer
    db2=1/m*np.sum(dZ2) #error of the bias of the second layer being calculated by multiplying the inverse of m by the sum of the error of the second layer
    dZ1=W2.T.dot(dZ2)*deriv_ReLU(Z1) #error of the first layer which is equal to the dot product of the transposed weight of the second layer and the error of the second layer multiplied by the derivative of the ReLU function of the first layer
    dW1=1/m*dZ1.dot(X.T) #error of the weights of the first layer which is equal to the inverse of m multiplied by the dot product of the error of the first layer and the transpose of the input layer
    db1=1/m*np.sum(dZ1) #error of the bias of the first layer equal to the inverse of m multiplied by the sum of the errors of the first layer

    return dW1,db1,dW2,db2

#UPDATING PARAMETERS
def update_paras( W1, b1, W2, b2, alpha, dW1, db1, dW2, db2): #function to update parameters
    W1=W1-alpha*dW1 #sets the new weights for the first layer as the weights of the first layer minus the product of the learning rate and the error of the weights of the first layer
    b1=b1-alpha*db1 #sets the new bias for the first layer as the bias of the first layer minus the product of the learning rate and the error of the bias of the first layer
    W2=W2-alpha*dW2
    b2=b2-alpha*db2

    return W1,b1,W2,b2

#GRADIENT DESCENT OR TRAINING
def get_predictions(A2): #function to fetch the predictions
    return np.argmax(A2, 0) #used to get the index of the highest value(probability) in A2

def get_accuracy(predictions, Y): #function to calculate accuracy
    print(predictions, Y)
    return np.sum(predictions==Y)/Y.size #compares the predictions with the labels and adds up the boolean values and divides it by the total number of labels to calculate accuracy

def gradient_descent(X, Y, iterations, alpha): #gradient descent function
    W1,b1,W2,b2=params() #get all these parameters with random values using the function we defined
    for i in range(iterations): #for loop with user specified number of iterations
        Z1,A1,Z2,A2=forward_prop(W1,b1,W2,b2,X) #get layer values using the forward propagation function
        dW1,db1,dW2,db2=back_prop(Z1,A1,Z2,A2,W2,X,Y) #get the errors in the layers using the backward propagation function
        W1,b1,W2,b2=update_paras(W1,b1,W2,b2,alpha,dW1,db1,dW2,db2) #update the parameters by subtracting the product of the learning rate and the error
        if (i%100==0): #print iteration number and accuracy every 100 iterations
            print('iteraion - ',i)
            predictions=get_predictions(A2)
            print('accuracy - ', get_accuracy(predictions, Y))
        
    return W1,b1,W2,b2

W1,b1,W2,b2=gradient_descent(X_train, Y_train, 800, 0.10) #arguement to run the gradient descent function on train data with 500 iterations and 0.10 as the learning rate

#TESTING
def make_predictions(X, W1, b1, W2, b2): #function to make predictions
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X) # _ are used as throwaway variables to ignore the first three values returned by forward_prop() and keep only A2
    predictions = get_predictions(A2) 
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_dev[:, index, None] # X_dev[:, index] extracts the index-th column (one image). :, None reshapes it into a column vector (ensuring it remains 2D).
    prediction = make_predictions(X_dev[:, index, None], W1, b1, W2, b2)
    label = Y_dev[index] #Retrieves the correct label from Y_dev
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255 #Prepares the image for display
    #Displays the image using Matplotlib
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







