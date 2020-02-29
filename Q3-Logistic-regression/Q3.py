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

x_dim = x_train.shape[1]
x_col = (x_train.shape[0])

# bias terms for training and testing set
train_bias = np.ones((x_train.shape[0],1))
test_bias = np.ones((x_test.shape[0],1))

# adding Bias term into dataset
x_train_bias = np.append(train_bias,L_x_train,axis=1)
x_test_bias = np.append(test_bias,L_x_test,axis=1)

# identity matrix
I = np.identity(x_dim + 1)
I[0][0] = 0

lmd_set = np.append(np.arange(1, 10), np.arange(10, 101, 5))

def sigmoid(z):
    return 1/(1+np.exp(-z))

def train(x,y,lmd):
    e = 0.000000001  
    W = np.zeros((x_dim + 1, 1))
    margin = 1
    iteration = 0
    while margin > e:  
        z = np.dot(x,W)
        mu = sigmoid(z)

        #weights for regularization term
        W_reg = W.copy()
        W_reg[0][0] = 0

        # gradient decent
        grad = x.T.dot(mu - y) + lmd*W_reg

        # hessian
        S = np.diag(mu[:, 0]) * np.diag(1 - mu[:, 0])
        H = (x.T.dot(S)).dot(x) + lmd*I
        
        #descent value 
        d_val = np.linalg.inv(H).dot(grad)
        # update weights
        W = W - d_val
        # model converge when margin reach critical value
        margin = np.dot(d_val.T,d_val)
    return W  


def predict(x,w):
    return 1 if np.dot(w.T,x) > 0 else 0

def error_rates(x,y,w):
    error = 0
    for i in range(x.shape[0]):
        if predict(x[i],w) != y[i] :
            error += 1
    return error/x.shape[0]

train_error = []
test_error = []

start_time = time.time()
for i in range(len(lmd_set)):
    W_train = train(x_train_bias,y_train,lmd_set[i])
    train_error.append(error_rates(x_train_bias,y_train,W_train))
    test_error.append(error_rates(x_test_bias,y_test,W_train))
    if i == 0 or i == 9 or i == 27:
        print("when Lambda is "+ str(lmd_set[i])+" training error rate is " + str(error_rates(x_train_bias,y_train,W_train)) )
        print("when Lambda is "+ str(lmd_set[i])+" testing error rate is " + str(error_rates(x_test_bias,y_test,W_train)) )

print("Test different Lambda finished in %04.3f seconds" % (time.time() - start_time))

# save result
result_data = {
    "lambda": lmd_set,
    "train_err_rate": train_error,
    "test_err_rate": test_error
}
np.save("Logistic regression error rate.npy",result_data)

# plot error rate
fig = plt.figure(1)
plt.plot(lmd_set,train_error,color='b',label='train error rate')
plt.plot(lmd_set,test_error,color='r',label='test error rate')
plt.xlabel('Lambda')
plt.ylabel('Error Rates')
plt.legend(loc=0)
fig.tight_layout()
plt.savefig('Q3_Error rate -- Logistic regression.png')
plt.show()








