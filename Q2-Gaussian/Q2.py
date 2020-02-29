import numpy as np
from scipy import io
import matplotlib.pyplot as plt

# Load Data
mat = io.loadmat('../spamData.mat')
x_train = mat['Xtrain']  #shape (3065, 57) 
y_train = mat['ytrain']
x_test = mat['Xtest']  #shape (1536, 57)
y_test = mat['ytest']
x_dim = x_train.shape[1]  # features dimensions - 57

# Log-transform to preprocess data
L_x_train = np.log(x_train + 0.1)
L_x_test = np.log(x_test + 0.1)

# Estimate class prior using ML
count_train = y_train.shape[0]
count_y1 = 0
for n in y_train:
    if n == 1:
        count_y1 += 1
lambda_prior = count_y1 / count_train 


# Cascade label y to x
xy_train = np.append(L_x_train, y_train, axis = 1)

# spam dataset 
spam_train = xy_train[xy_train[:,-1] > 0]
x_spam_train = spam_train[:,:-1]

# mean for each feature in spam dataset
x_mean_spam = np.mean(x_spam_train, axis = 0)

# varicance for each feature in spam dataset
x_var_spam =np.var(x_spam_train, axis = 0)

# ham dataset
ham_train = xy_train[xy_train[:,-1] < 1]
x_ham_train = ham_train[:,:-1]

# mean for each feature in ham dataset
x_mean_ham = np.mean(x_ham_train, axis = 0)

# varicance for each feature in spam dataset
x_var_ham =np.var(x_ham_train, axis = 0)

def gaussian(x,mean,var):
    return np.exp(-0.5*(x - mean)**2/var)/ np.sqrt(2*np.pi* var)

# probability is spam
def p_spam(x):
    p_spams = 0
    for i in range(x_dim):
        p_spams += np.log(gaussian(x[i],x_mean_spam[i],x_var_spam[i]))
    p_spams = np.log(lambda_prior) + p_spams
    return p_spams

# probability is ham
def p_ham(x):
    p_hams = 0
    for i in range(x_dim):       
            p_hams += np.log(gaussian(x[i],x_mean_ham[i],x_var_ham[i]))       
    p_hams = np.log(1-lambda_prior) + p_hams
    return p_hams


# predict
def predict(x):
    np.seterr(divide = 'ignore') 
    return 1 if p_spam(x) > p_ham(x) else 0

# validation for test dataset
def error_rates(L_x_test,y_test):
    errors = 0
    count_test = L_x_test.shape[0]
    for i in range(count_test):
        if predict(L_x_test[i,:]) != y_test[i,:]:
            errors += 1
    return errors/count_test

print("Training error rates: " + str(error_rates(L_x_train, y_train)))
print("Testing error rates: " + str(error_rates(L_x_test, y_test)))

