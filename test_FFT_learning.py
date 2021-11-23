import numpy as np
import os
# add bin folde directory for GPU computing
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
from tensorflow.python.keras import initializers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

# a function to see if a Neural Network (NN) can predict it
def func(x):
    return 2*np.sin(2*np.pi*x)+3*np.cos(2*np.pi*x) + np.exp(x)

# number of samples
num_samples = 100
num_chanels = 1
x = np.random.random((num_samples,num_chanels))
y = func(x)

class  DFT_layer(Layer):
    def __init__(self,units):
        super(DFT_layer,self).__init__()
        self.Length = 1
        self.units = units
        self.var1 = np.arange(self.units)
    def build(self,input_shape):        
        self.w = self.add_weight(shape=(2*self.units,1),
            initializer='random_normal',
            trainable=True)
    def call(self,inputs):
        temp_var1 = tf.math.cos(2*np.pi*inputs*self.var1/self.Length)
        temp_var2 = tf.math.sin(2*np.pi*inputs*self.var1/self.Length)
        output_layer = tf.matmul(temp_var1,self.w[0:self.units]) + tf.matmul(temp_var2,self.w[self.units:2*self.units])
        return output_layer


def myNet(num_freq,learning_rate=1e-3):
    layer0 = keras.layers.Input((1))
    layer1 = Dense(1,activation='linear')(layer0)
    layer2_class = DFT_layer(num_freq)
    layer2 = layer2_class(layer1)
    #layer2 = Dense(20,activation='relu')(layer1)
    outputs =  Dense(1,activation='linear')(layer2)
    model = keras.Model(inputs=layer0, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # 4.1 -- The loss is chosen to be mean absolute error
    model.compile(optimizer=optimizer, loss=['mean_absolute_error'], metrics=['mean_absolute_error'])
    # 4.2 -- Just print the model's summary for information
    model.summary()
    return model


model = myNet(10)

#model.compile(optimizer='adam', loss=['mean_absolute_error'])

model.fit(x, y, epochs=500, batch_size=32,verbose=2,validation_split=0.3)

import matplotlib.pyplot as plt

plt.plot(x[:,0],y[:,0],'o')
x_random = np.random.random((100,num_chanels))
y_predict = model.predict(x_random)
plt.plot(x_random[:,0],y_predict[:,0],'*')
plt.show()
print('End!')
