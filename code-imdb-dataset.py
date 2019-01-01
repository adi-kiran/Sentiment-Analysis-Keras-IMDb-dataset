import numpy as np
import keras
#importing the required libraries (numpy and keras)
#keras uses tensorflow as backend and it need not be imported separately

#importing data from keras-imdb sentiment classification dataset
from keras.datasets import imdb

#importing specific functions/mini-libraries from keras to build and design the layers of our model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

#declaring max_words per review to be analyzed
max_words =500

#using a 50-50% train-test dataset
(x_train, y_train),(x_test, y_test)=imdb.load_data(num_words=max_words,seed=50)

#length of input test and train sequences
print('length of x_train sequences',len(x_train))
print('length of x_test sequences',len(x_test))

#number of sentiments to be classified
no_class = np.max(y_train) + 1
print(no_class, 'classes')

#format the text to a readable matrix that can be analyzed by the system
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train)
x_test = tokenizer.sequences_to_matrix(x_test)

#x test and train has size 25000,500 because 25000 test and train values and max words given as 500
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

#making y a binary matrix where 1 represents that a particular sentiment is the chosen and 0 represents that the sentiment does not manifest/not chosen
y_train = keras.utils.to_categorical(y_train, no_class)
y_test = keras.utils.to_categorical(y_test, no_class)

#y test and train has size of 25000,2 because 2 sentiments to classify and 25000 values for test and train each
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

#creating our sequential model for analyzing the data
model = Sequential()

#hidden layer having 300 neurons(when tested multiple times with 700 and 500 neurons the accuracy was similar to that of 300 neurons)
model.add(Dense(300, input_shape=(max_words,)))
#it is activated by RelU function
model.add(Activation('relu'))
#dropout is to prevent over fitting of data
model.add(Dropout(0.2))

#output layer having same number of neurons as the sentiments(2)
model.add(Dense(no_class))
#softmax is used as activation/classification function
model.add(Activation('softmax'))

#compiling the model with categorical crossentropy as cost function and adam optimizer taking accuracy as metrics
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#fitting data for testing using batches of 20 values
batch_size = 20
epochs = 5
history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_split=0.1)

#Evaluating the test data on the model to get accuracy
score = model.evaluate(x_test, y_test,batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])

#accuracy obtained about 0.82 over multiple evaluations.