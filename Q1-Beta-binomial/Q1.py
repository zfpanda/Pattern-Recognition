import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import time

# Run this script will generate training and testing error rates 
# data saved into Q1_result.pyp and plot saved into Q1_Error rate -- Beta-bernoulli Naive Bayes.png

# Load Data
mat = io.loadmat('../spamData.mat')
x_train = mat['Xtrain']  #shape (3065, 57) 
y_train = mat['ytrain']
x_test = mat['Xtest']  #shape (1536, 57)
y_test = mat['ytest']
x_dim = x_train.shape[1]  # features dimensions 57

# Binarization to pre process data 
B_x_train = np.where(x_train > 0, 1, 0)
B_x_test = np.where(x_test > 0, 1, 0)

# Estimate class prior using ML
count_train = y_train.shape[0]
count_y1 = 0
for n in y_train:
    if n == 1:
        count_y1 += 1
lambda_prior = count_y1 / count_train 

# store No. of 1s appear per feacture for both spam and ham dataset
counts = np.zeros(shape = (2,x_dim))

# Cascade label y to x
xy_train = np.append(B_x_train, y_train, axis = 1)

# spam dataset 
spam_train = xy_train[xy_train[:,-1] > 0]
x_spam_train = spam_train[:,:-1]

# count 1s per feature column in spam dataset
count_1s = np.count_nonzero(x_spam_train, axis = 0)
counts[0] = count_1s

# ham dataset
ham_train = xy_train[xy_train[:,-1] < 1]
x_ham_train = ham_train[:,:-1]

# count 1s per feature in ham dataset
count_0s = np.count_nonzero(x_ham_train, axis = 0)
counts[1] = count_0s

# store feature likelihood probability per feature for both spam and ham dataset
theta = np.zeros(shape = (2,x_dim))

def train(alpha):
    for i in range(x_dim):
        theta[0,i] = (count_1s[i] + alpha)/(count_y1 + 2*alpha)
        theta[1,i] = (count_0s[i] + alpha)/((count_train - count_y1) + 2*alpha)

# probability is spam
def p_spam(x):
    p_spams = 0
    for i in range(x_dim):
        if x[i] > 0:
            p_spams += np.log(theta[0,i])
        else:
            p_spams += np.log(1 - theta[0,i])
    p_spams = np.log(lambda_prior) + p_spams
    return p_spams

# probability is not spam
def p_ham(x):
    p_hams = 0
    for i in range(x_dim):
        if x[i] > 0:
            p_hams += np.log(theta[1,i])
        else:
            p_hams += np.log(1 - theta[1,i])
    p_hams = np.log(1-lambda_prior) + p_hams
    return p_hams

# predict
def predict(x):
    return 1 if p_spam(x) > p_ham(x) else 0

# validation for test dataset
def error(x,y,alpha):
    errors = 0
    data_size = x.shape[0]
    train(alpha)
    for i in range(data_size):
        if predict(x[i,:]) != y[i,:]:
            errors += 1
    return errors/data_size

# train error and test error in alpha set 
alpha_set = np.arange(0,100.5,0.5)
train_err = np.empty(len(alpha_set))
test_err = np.empty(len(alpha_set))

def error_alpha_set():
    for i in range(len(alpha_set)):
        train_err[i] = error(B_x_train,y_train,alpha_set[i])
        test_err[i] = error(B_x_test,y_test,alpha_set[i])
        if alpha_set[i] == 1 or alpha_set[i] == 10 or alpha_set[i] == 100 :
            print("when Alpha is "+ str(alpha_set[i]) +" Train error rate is " + str(train_err[i]))
            print("when Alpha is "+ str(alpha_set[i]) +" Test error rate is " + str(test_err[i]))

    
start_time = time.time()
error_alpha_set() 
print("Training and testing for alpha(0 ~ 100 / step0.5) completed in %04.3f seconds!" % (time.time() - start_time))

# save data
result_data = {
    "alpha": alpha_set,
    "train_err_rate": train_err,
    "test_err_rate": test_err
}

np.save("Beta-bernoulli error rate.npy",result_data)


# plot error rate
fig = plt.figure(1)
plt.plot(alpha_set,train_err,color='b',label='train error rate')
plt.plot(alpha_set,test_err,color='r',label='test error rate')
plt.xlabel('Alpha')
plt.ylabel('Error Rates')
plt.title('Error rate -- Beta-bernoulli Naive Bayes')
plt.legend(loc=0)
fig.tight_layout()
plt.savefig('Q1_Error rate -- Beta-bernoulli Naive Bayes.png')
plt.show()



























