#class Neural Network
import numpy as np

class NeuralNetwork:
    def __init__(self):
        np.random.seed(1)
        return
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self,x):
        return x*(1-x)

    def hyptan(self,x):
        return np.tanh(x),np.cosh(x)
    
    def hyptan_derivative(self,x):
        return 1/x**2
    
    def train(self,x,y,iterations,learning_rate,batch_size = 500):
        a,b = x.shape
        self.weight1 = np.random.random((b,50)) - 1
        self.weight2 = np.random.random((50,50)) - 1
        self.weight3 = np.random.random((50,1)) - 1
        for i in range(iterations):
            ids = np.random.choice(a, batch_size, replace=False)
            x_batch = x[ids]
            y_batch = y[ids]
            self.gradient_descent(x_batch,y_batch,learning_rate)
            
    def gradient_descent(self,x,y,learning_rate):
        layer1 = np.dot(x,self.weight1)
        #print layer1.shape
        layer1act,layer1cosh = self.hyptan(layer1)
        #print layer1act.shape,layer1cosh.shape
        layer2 = np.dot(layer1act,self.weight2)
        #print layer2.shape
        layer2act = self.sigmoid(layer2)
        #print layer2act.shape
        layer3 = np.dot(layer2act,self.weight3)
        #print layer3.shape
        layer3act = self.sigmoid(layer3)
        #print y.shape
        #print layer3act.shape
        #print layer3act.shape
        y = y.reshape([len(y),1])
        error = y - layer3act
        layer3dev = np.dot(layer2act.T,error * self.sigmoid_derivative(layer3act))
        self.weight3 = self.weight3 - learning_rate*layer3dev
        layer2dev = np.dot(np.dot(error * self.sigmoid_derivative(layer3act),self.weight3.T).T,self.sigmoid_derivative(layer2act)*layer1act)
        self.weight2 = self.weight2 - learning_rate*layer2dev
        layer1dev = np.dot(self.weight1,np.dot(np.dot(np.dot(error * self.sigmoid_derivative(layer3act),
                                                                            self.weight3.T)*self.sigmoid_derivative(layer2act),
                                                                    self.weight2).T,
                                                            self.hyptan_derivative(layer1cosh)))
        # np.dot(x.T,self.hyptan_derivative(layer1cosh))
        #print layer1dev.shape
        #print self.weight1.shape
        self.weight1 = self.weight1 - learning_rate*layer1dev    
    
    def predict(self,x):
        layer1 = np.dot(x,self.weight1)
        layer1act,layer1cosh = self.hyptan(layer1)
        layer2 = np.dot(layer1act,self.weight2)
        layer2act = self.sigmoid(layer2)
        layer3 = np.dot(layer2act,self.weight3)
        layer3act = self.sigmoid(layer3)
        pred = []
        for i in layer3act:
            if i>0.4:
                pred.append(0)
            else:
                pred.append(1)
        return pred
    
    def accuracy(self,a,b):
        a = np.asarray(a)
        b = np.asarray(b)
        return np.mean(a==b)*100