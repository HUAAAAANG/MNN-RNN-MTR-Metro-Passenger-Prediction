import keras
from keras.models import Sequential
from keras.layers import Dense
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np

#data = genfromtxt('Metro_Interstate_Traffic_Volume.CSV', delimiter=',', skip_header=1, usecols=range(0,7))
#data = genfromtxt('Metro_Interstate_Traffic_Volume_Reduced_Reduced.CSV', delimiter=',', skip_header=0, usecols=range(0,7))
data = genfromtxt('Reduced.CSV', delimiter=',', skip_header=0, usecols=range(0,7))
x_input = data[:,0:6]
y_expect = data[:,6]
N = len(y_expect)
        
for i in range(0,7):
    x_input[:,i:i+1] = x_input[:,i:i+1]/max(x_input[:,i:i+1])

#for i in range(0,7):
#    x_input[:,i:i+1] = (x_input[:,i:i+1]/max(x_input[:,i:i+1]))*2-1
    
#rescale data for output
y_expect = y_expect/10000

#disrupt the order of input and output

order_shuffled = list(range(N))
np.random.shuffle(order_shuffled)
x_input = x_input[order_shuffled,:]
y_expect = y_expect[order_shuffled]

# devide the data into 3 parts: training, validation and test
Tp = int(0.8*N)
trainX = x_input[0:Tp,:]
trainY = y_expect[0:Tp]

 # prepare the test set
testX = x_input[Tp:N,:]
testY = y_expect[Tp:N]


n1=6
n2=2
n3=1
model = Sequential()
keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
#keras.initializers.RandomUniform(minval=-0.408, maxval=0.408)
model.add(Dense(units=n2, input_dim=n1, activation='sigmoid'))
model.add(Dense(units=n3, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
#model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(learning_rate=0.1), metrics=['mean_squared_error'])
#model.compile(loss='mean_squared_error', optimizer='nadam', metrics=['mean_squared_error'])

train_history = model.fit(x=trainX, y=trainY, validation_split=0.25, epochs=1000, batch_size=32, verbose=2)
trainPredict = model.predict(trainX) # predicted output on train & validation set
testPredict = model.predict(testX) # predicted output on test set


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train], '-')
    plt.plot(train_history.history[validation], '--')
    plt.title('Train History', fontsize=16)
    plt.tick_params(labelsize=16)
    plt.ylabel(train, fontsize=18)  
    plt.xlabel('Epoch', fontsize=18)
    plt.legend(['Train', 'Validate'], loc='best', fontsize=16)
    plt.ylim([0,0.05])
    plt.show()
show_train_history(train_history, 'mean_squared_error', 'val_mean_squared_error')

test_MSE = np.mean((testPredict-testY)**2)
print('test_MSE =', test_MSE)
test_MAPE = np.mean(np.abs(testPredict-testY)/testY)
print('test_MAPE =', test_MAPE)

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
