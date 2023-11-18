import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
step = 12 # number of hours

#load and prepare the data
#data = genfromtxt('Metro_Interstate_Traffic_Volume.CSV', delimiter=',', skip_header=1, usecols=range(0,7))
#data = genfromtxt('Metro_Interstate_Traffic_Volume_Reduced_Reduced.CSV', delimiter=',', skip_header=0, usecols=range(0,7))
data = genfromtxt('Reduced.CSV', delimiter=',', skip_header=0, usecols=range(0,7))

x_input = data[:,0:6]
y_expect = data[:,6]
N = len(y_expect)

#rescale data for input
for i in range(0,7):
    x_input[:,i:i+1] = (x_input[:,i:i+1]/max(x_input[:,i:i+1]))*2-1

#rescale data for output

y_expect = y_expect/10000

#disorder
order_shuffled = list(range(N))
np.random.shuffle(order_shuffled)
x_input = x_input[order_shuffled,:]
y_expect = y_expect[order_shuffled]

# devide the data into 3 parts: training, validation and test
Tp = int(0.8*N) # first 80% of days for training & validation
train_x = x_input[0:Tp,:]
train_y = y_expect[0:Tp]
trainX = np.array([train_x[i:i+step,:] for i in range(len(train_x)-step)])
trainY = np.array(train_y[step:])

# prepare the test set
test_x = x_input[Tp:N,:]
test_y = y_expect[Tp:N]
testX = np.array([test_x[i:i+step,:] for i in range(len(test_x)-step)]) # input
testY = np.array(test_y[step:]) # expected output

# create the recurrent neural network
model = Sequential()
kk = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
#model.add(SimpleRNN(units=5, input_shape=(step,6), activation="sigmoid")) # simple RNN
model.add(LSTM(units=3, input_shape=(step,6))) # LSTM
model.add(Dense(units=1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
model.summary()

# train the RNN: 0.8*0.25=20% for validation, 0.8-0.2=60% for training
train_history = model.fit(x=trainX, y=trainY, validation_split=0.25, epochs=1000, batch_size=32, verbose=2)
trainPredict = model.predict(trainX) # predicted output on train & validation set
testPredict = model.predict(testX) # predicted output on test set

# plot the metrics to show the training performance
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train], '-')
    plt.plot(train_history.history[validation], '--')
    plt.title('Train History', fontsize=16)
    plt.tick_params(labelsize=16)
    plt.ylabel(train, fontsize=18)  
    plt.xlabel('Epoch', fontsize=18)
    plt.legend(['Train', 'Validate'], loc='best', fontsize=16)
    plt.ylim([0.02, 0.05])
    plt.show()
show_train_history(train_history, 'mean_squared_error', 'val_mean_squared_error')

# visualize the result
for i_plot in [1,2]:
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)  
    plt.figure(figsize=(40,20))
    plt.plot(range(0,Tp), trainY*10000, 'b')
    plt.plot(range(0,Tp), trainPredict*10000, 'r')
    plt.legend(['Actual Passangers', 'Predicted Passangers'], loc='best', fontsize=20)  
    
    plt.plot(range(Tp,N), testY*10000, 'b')
    plt.plot(range(Tp,N), testPredict*10000, 'r')
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Number of passangers', fontsize=20)
    if i_plot==2:
        plt.xlim([Tp, N])
    if i_plot==1:
        plt.title('Whole time period', fontsize=20)
    else:
        plt.title('Test period', fontsize=20)
    plt.show()

# output the MSE for test sets
test_MSE = np.mean((testPredict-testY)**2)
print('test_MSE =', test_MSE)