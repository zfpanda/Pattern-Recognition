import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import time

# Load Data
mat = io.loadmat('../spamData.mat')
x_train = mat['Xtrain']  #shape (3065, 57) 
y_train = mat['ytrain']
x_test = mat['Xtest']  #shape (1536, 57)
y_test = mat['ytest']

# Log-transform to pre process data
L_x_train = np.log(x_train + 0.1)
L_x_test = np.log(x_test + 0.1)

dataSize = L_x_train.shape[0]

def euclideanDistance(xi):
    diff = diff = np.tile(xi,(dataSize,1)) - L_x_train
    sqdiff = diff**2
    squareDist = np.sum(sqdiff,axis = 1)
    return np.sqrt(squareDist)

def KNN_classfier(k,input):   
    dist = euclideanDistance(input)
    # Sorting the distances,range from small to big, renturn index
    sortedDistIndex = np.argsort(dist)
    
    voteLable = []
    # get first K value near the input
    for i in range(k):
        voteLable.append(y_train[sortedDistIndex[i]])
    # check the higher occurence of label and make prediction
    if voteLable.count(1) > voteLable.count(0):
        return 1
    else:
        return 0

def error_rates(input,y,k):
    error = 0
    for i in range(input.shape[0]):
        if KNN_classfier(k,input[i]) != y[i]:
            error += 1
    return error / input.shape[0]

k_set = np.append(np.arange(1, 10), np.arange(10, 101, 5))

train_error = []
test_error = []

start_time = time.time()
print("start!")

for i in range(len(k_set)):
    train_error.append(error_rates(L_x_train,y_train,k_set[i]))
    test_error.append(error_rates(L_x_test,y_test,k_set[i]))
    if i == 0 or i == 9 or i == 27:
        print("when K is "+ str(k_set[i])+" training error rate is " + str(error_rates(L_x_train,y_train,k_set[i])) )
        print("when K is "+ str(k_set[i])+" testing error rate is " + str(error_rates(L_x_test,y_test,k_set[i])) )
print("k set finished in %04.3f seconds" % (time.time() - start_time))

# save results
result_data = {
    "k_set": k_set,
    "train_error_rate": train_error,
    "test_error_rate": test_error
}
np.save("KNN_error_rate.npy",result_data)


# plot error rate
fig = plt.figure()
plt.plot(k_set,train_error,color='b',label='train error rate')
plt.plot(k_set,test_error,color='r',label='test error rate')
plt.xlabel('K')
plt.ylabel('Error Rates')
plt.legend(loc=0)
fig.tight_layout()
plt.savefig('Q4_Error rate -- KNN.png')
plt.show()