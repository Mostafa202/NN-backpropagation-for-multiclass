import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BP:
    def __init__(self,layers_units,lr,epochs,weights=None):
        self.lr=lr
        self.epochs=epochs
        self.layers_units=layers_units
        self.layers_count=len(layers_units)
        #define weights matrices
        self.weights=weights
        self.weights_matrices=[]
        if self.weights==None:
            for l in range(self.layers_count-1):
                self.num_units_first=self.layers_units[l]
                self.num_units_second=self.layers_units[l+1]
                #+1 means bias
                self.weights_matrices.append(np.random.rand(self.num_units_second,self.num_units_first+1))
        else:
            self.weights_matrices=self.weights
            
            
    def active_func_softmax(self,prob):
        """ Softmax function is not really a stable one, if you implement this using 
        python you will frequently get nan error 
        due to floating point limitation in NumPy ,constant may be -max(z)"""

        return (np.exp(prob)/np.sum(np.exp(prob))).T
    def active_func_sigmoid(self,prob):
        return (1.0/(1.0+np.exp(-prob))).T
        
    def propagation_func(self,x,w):
        self.prob=w.dot(np.matrix(x).T)
        return self.prob
    
    def feed_forward(self,x):
        self.y_hats_for_each_layer=[np.matrix(x)]
        for i in range(self.layers_count-1):
            self.train_sample=self.y_hats_for_each_layer[i]
            if i!=self.layers_count-2:
                self.y_hat_l=self.active_func_sigmoid(self.propagation_func(self.train_sample,self.weights_matrices[i]))
            else:
                self.y_hat_l=self.active_func_softmax(self.propagation_func(self.train_sample,self.weights_matrices[i]))
            self.y_hat=self.y_hat_l
            # insert bias in each layer except the output layer
            if i!=self.layers_count-2:
                self.y_hat=np.insert(self.y_hat,0,1)
            self.y_hats_for_each_layer.append(self.y_hat)
        return self.y_hats_for_each_layer
    
    def back_and_adjust(self,x,y):
        self.hats=self.feed_forward(x)
        #calc_deltas in reverse        
        #self.derive=np.multiply(self.hats[-1],(1-self.hats[-1]))
        self.delta=(y-self.hats[-1])
        self.deltas=[self.delta.T]
        j=0
        for i in range(len(self.hats)-2,0,-1):
            #remove column and transpose for weights matrices to avoid delta for bias units
            #w.T*delta
            self.w=np.delete(self.weights_matrices[i],0,1)
            #mul_hat
            self.mul_hat=np.delete(self.hats[i],0,1)
            self.mul_hat2=np.multiply(self.mul_hat,(1-self.mul_hat))
            self.deltas.append(np.multiply(self.mul_hat2.T,(self.w.T).dot(self.deltas[j])))
            j+=1
        #adjustment for weights
        #w:=w+lr*delta*o
        self.deltas=self.deltas[::-1]
       

        #self.mul_hats=self.hats[1:]
        for k in range(len(self.weights_matrices)):
            
            self.weights_matrices[k]+=self.lr*(self.deltas[k].dot(self.hats[k]))
        
            
    def training(self,x,y):
        for p in range(self.epochs):
            self.check=True
            print('================= epoch(',(p+1),')===============')
            print('W:',self.weights_matrices)
            for i in range(len(x)):
                self.rand_num=np.random.randint(len(x))
                self.instance_x=x[self.rand_num]
                self.instance_y=y[self.rand_num]
                self.hats_end=self.feed_forward(self.instance_x)
                self.error=np.abs(self.instance_y-self.hats_end[-1])
                if self.error.any()>0.0001:
                    self.back_and_adjust(self.instance_x,self.instance_y)
                    print('Error: ',self.error)
                    self.check=True

                else:
                    self.check=False
            if self.check==False:
                break
            
    def testing(self,x_test):
        self.y_pred=[]
        for i in range(len(x_test)):
            self.x_i=x_test[i]            
            self.y_pred.append(np.argmax(self.feed_forward(self.x_i)[-1]))
        return self.y_pred


dataset=pd.read_csv('project_data.csv')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import *
###s=StandardScaler()
###x=s.fit_transform(x)
lb=LabelEncoder()
y=lb.fit_transform(y)

one=OneHotEncoder(categorical_features=[0])
y=one.fit_transform(y.reshape(-1,1)).toarray()

x=np.append(np.ones((len(x),1)),x,axis=1)

from sklearn.model_selection import *

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

b=BP([7,3,2,3],0.1,50)
            
b.training(x_train,y_train)

predict=b.testing(x_test)

com_y=np.argmax(y_test,axis=1)

accuracy=(np.sum(np.array(predict)==com_y)/len(com_y))*100
print('accuracy:',accuracy,' %')

