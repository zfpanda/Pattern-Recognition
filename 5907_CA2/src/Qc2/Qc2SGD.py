import numpy as np
import struct
import matplotlib.pyplot as plt
import gzip
import sys
import time
from sklearn.preprocessing import PolynomialFeatures
import scipy.special as spc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
from scipy import interp
from itertools import cycle

def loadImageSet(filename):
    with gzip.open(filename, 'rb') as binfile:
        buffers = binfile.read()    
        head = struct.unpack_from('>IIII', buffers, 0) # take the first 4 integers and return a tuple
    
        offset = struct.calcsize('>IIII')  # position to the initial location of data
        imgNum = head[1]
        width = head[2]
        height = head[3]
    
        bits = imgNum * width * height  # data--60000*28*28
        bitsString = '>' + str(bits) + 'B'  # fmt：'>47040000B'
    
        imgs = struct.unpack_from(bitsString, buffers, offset) # retrive data，return as tuple
    
        binfile.close()
        imgs = np.reshape(imgs, [imgNum, width * height]) # reshape to [60000,784] numpy array
 
        return imgs
 
 
def loadLabelSet(filename):
    with gzip.open(filename, 'rb') as binfile:
        buffers = binfile.read()
    
        head = struct.unpack_from('>II', buffers, 0) # read first 2 integers
    
        labelNum = head[1]
        offset = struct.calcsize('>II')  # position to the initial place of label data
    
        numString = '>' + str(labelNum) + "B" # fmt：'>60000B'
        labels = struct.unpack_from(numString, buffers, offset) # get label data
    
        binfile.close()
        labels = np.reshape(labels, [labelNum]) # reshape to one dimension array
    
        return labels

x_train_raw = loadImageSet('../train-images-idx3-ubyte.gz')
x_test_raw = loadImageSet('../t10k-images-idx3-ubyte.gz')

y_train = loadLabelSet('../train-labels-idx1-ubyte.gz')
y_test = loadLabelSet('../t10k-labels-idx1-ubyte.gz')

# mean of the training digits(mean of per feature)
meanVal = np.mean(x_train_raw,axis = 0)

# subtract the mean from each training data point and test data point, zero centering
x_train_z = x_train_raw - meanVal
x_test_z = x_test_raw - meanVal

# Compute the covariance matrix using all the training images
covMat = np.cov(x_train_z,rowvar = 0)

# get eigenvalue and eigenvector from covirance matrix
featValue,featVec = np.linalg.eig(covMat)

# sort the eigenvalue in ascending order and return index
index = np.argsort(featValue)


# Handwritten digit image reconstruction

n_30_index=index[-1:-31:-1] #get indexes of the 30 largest eigenvalues
n_30_featVec=featVec[:, n_30_index] # get eigenvectors matrix corresponding to the largest k dimension eigenvalues

n_20_index=index[-1:-21:-1] #get indexes of the 20 largest eigenvalues
n_20_featVec=featVec[:, n_20_index] # get eigenvectors matrix corresponding to the largest k dimension eigenvalues


# Die=mension reduction to 30 for train and test set
X_train = np.dot(x_train_z,n_30_featVec) /255
X_test = np.dot(x_test_z,n_30_featVec) / 255

# Die=mension reduction to 20 for train and test set
X_train_20 = np.dot(x_train_z,n_20_featVec) /255
X_test_20 = np.dot(x_test_z,n_20_featVec) / 255

poly2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly2.fit_transform(X_train)
X_test_poly2 = poly2.fit_transform(X_test)

poly3 = PolynomialFeatures(degree=3)
X_train_poly3 = poly3.fit_transform(X_train_20)
X_test_poly3 = poly3.fit_transform(X_test_20)
print(X_test_poly3.shape)

def adjustment_label(labelset):
    new_labelset = []
    for i in labelset:
        new_label = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        new_label[i] = 0.99
        new_labelset.append(new_label)
    return new_labelset

train_label = adjustment_label(y_train)
test_label = adjustment_label(y_test)

y_train_10 = (np.array(train_label))
y_test_10 = (np.array(test_label))

w_ploy2 = np.zeros((496,10))
b = np.ones(10)
lr = 0.001

for j in range(10):
    for i in range(60000):
        d = X_train_poly2[i]
        y = y_train_10[i]
        aa = np.dot(d,w_ploy2) + b
        y_ = aa/aa.sum()
        w_ploy2 = w_ploy2 + lr*np.dot(np.array([d]).T,np.array([(y-y_)]))
n = 0 
pre_ploy2 = []
for j in range(10000):
    pre = np.argmax(np.dot(X_test_poly2[j],w_ploy2))
    pre_ploy2.append(pre)
    label = np.argmax(y_test_10[j])
    if pre == label :
        n = n + 1
accuracy = n / 10000
print('Accuracy in 2nd order polynomial ',accuracy)

####polynomial in 3rd order
def sigmoid(z):
	return spc.expit(z)
w_poly3 = np.zeros((1771,10))

for j in range(10):
    for i in range(60000):
        d = X_train_poly3[i]
        y = y_train_10[i]
        sum = sigmoid(np.dot(d,w_poly3)+b)
        y_ = sum/sum.sum()
        w_poly3 = w_poly3 + lr*np.dot(np.array([d]).T,np.array([(y-y_)]))
n = 0 
pre_poly3 = []
for j in range(10000):
    pre = np.argmax(np.dot(X_test_poly3[j],w_poly3))
    pre_poly3.append(pre)
    label = np.argmax(y_test_10[j])
    if pre == label :
        n = n + 1
accuracy = n / 10000
print('Accuracy in 3rd order polynomial ',accuracy)


#Confusion Matrix

def plot_confusion_matrix(label,prediction,order):  
    confusion_mat=confusion_matrix(label,prediction)
    plt.imshow(confusion_mat,interpolation='nearest',cmap=plt.cm.Paired)
    plt.title('Confusion Matrix in SGD at '+str(order)+' order')
    plt.colorbar()
    tick_marks=np.arange(10)
    ind_array = np.arange(10)
    x_v, y_v = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x_v.flatten(), y_v.flatten()):        
        c = confusion_mat[y_val][x_val]
        plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=8, va='center', ha='center')
    plt.xticks(tick_marks,tick_marks)
    plt.yticks(tick_marks,tick_marks)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('Confusion Matrix in SGD at'+str(order)+' order Polynomial Regression.png')
    plt.show()


def plot_ROC(label,prediction,order):
    # Compute ROC curve for each class
    # Binarize the output
    Y_valid = label_binarize(label, classes=[i for i in range(10)])
    Y_pred = label_binarize(prediction, classes=[i for i in range(10)])


    # micro：Multi-class　　
    precision = precision_score(Y_valid, Y_pred, average='micro')
    recall = recall_score(Y_valid, Y_pred, average='micro')
    f1_scores = f1_score(Y_valid, Y_pred, average='micro')
    accuracy_scores = accuracy_score(Y_valid, Y_pred)
    print("Precision_score:",precision)
    print("Recall_score:",recall)
    print("F1_score:",f1_scores)
    print("Accuracy_score:",accuracy_scores)


    # roc_curve:（True Positive Rate , TPR）or（sensitivity）
    # x-coordinate：（False Positive Rate , FPR）

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve(Y_valid[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_valid.ravel(), Y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(10)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(10):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= 10

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(10), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve in SGD at '+str(order)+ ' order Polynomial Regression')
    plt.legend(loc="lower right")
    plt.savefig('ROC curve in SGD at '+str(order)+ ' order Polynomial Regression.png')
    plt.show()

#### 2nd order polynomial confusion matrix and ROC in SGD method
plot_confusion_matrix(y_test,pre_ploy2,2)
plot_ROC(y_test,pre_ploy2,2)

#### 3rd order polynomial confusion matrix and ROC in SGD method
plot_confusion_matrix(y_test,pre_poly3,3)
plot_ROC(y_test,pre_poly3,3)