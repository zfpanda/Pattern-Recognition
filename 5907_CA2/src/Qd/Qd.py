import numpy as np
import struct
import matplotlib.pyplot as plt
import gzip
import sys
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import time
import os
# import config_file as cfg_file
import datetime


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

def make_print_to_file(path='./'):
 
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)
 
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
 
        def flush(self):
            pass
 
    fileName = datetime.datetime.now().strftime('Date '+'%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)
 
    #############################################################
    # From here onwards, all content in print will write to log 
    #############################################################
    print(fileName.center(60,'*'))


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

# Dimension reduction to 30 for train and test set
x_train = np.dot(x_train_z,n_30_featVec) 
x_test = np.dot(x_test_z,n_30_featVec)

# Normalize the input dataset
x_train = x_train / 255
x_test = x_test / 255

poly2 = PolynomialFeatures(degree=2)
x_train_poly2 = poly2.fit_transform(x_train)
x_test_poly2 = poly2.fit_transform(x_test)


# one hot label
def adjustment_label(labelset):
    new_labelset = []
    for i in labelset:
        new_label = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
        new_label[i] = 1
        new_labelset.append(new_label)
    return new_labelset

train_label = adjustment_label(y_train)
test_label = adjustment_label(y_test)

y_train_10 = (np.array(train_label))
y_test_10 = (np.array(test_label))

######different eigen vectors in MSE Using Indicator Matrix (Primal) for linear and polynimial regression model####
def mnistUsingDiff_eigen_vectors():
    components = [30,35,40,50,60,80,100,150]
    accu = []
    accu_poly = []
    for m in components:
        pca = PCA(n_components=m)
        pca.fit(x_train_z)
        x_train_n = pca.transform(x_train_z)
        x_test_n = pca.transform(x_test_z)
        weights_n = ((np.linalg.inv(np.dot(x_train_n.T,x_train_n))).dot(x_train_n.T)).dot(y_train_10)
        if m < 100:
            poly2 = PolynomialFeatures(degree=2)
            X_train_poly2 = poly2.fit_transform(x_train_n)
            X_test_poly2 = poly2.fit_transform(x_test_n)   
            weights_poly_n = ((np.linalg.inv(np.dot(X_train_poly2.T,X_train_poly2))).dot(X_train_poly2.T)).dot(y_train_10)
        # predict test set output
        pre = np.dot(x_test_n,weights_n)
        pre_ploy = np.dot(X_test_poly2,weights_poly_n)
        # get the index of largest value and return as a list
        index = np.argmax(pre,axis = 1)
        index_poly = np.argmax(pre_ploy,axis = 1)
        # print(index.shape)
        # print(index_poly.shape)
        error = 0
        for i in range(len(index)):
            if index[i] != y_test[i]:
                error +=1
        accuray = str(round((1-error/len(index))*100,2))+'%'
        accu.append(accuray)
        if m < 100:
            error = 0
            for i in range(len(index_poly)):
                if index_poly[i] != y_test[i]:
                    error +=1
            accuray = str(round((1-error/len(index_poly))*100,2))+'%'
            accu_poly.append(accuray)
    print('For Eigen Vector in Dimensions '+ str(components).strip('[]'))
    print('The corresponding accuracies in   Linear Regression  model  are     '+  str(accu).strip('[]'))
    print('The corresponding accuracies in Polynomial Regression model are '+ str(accu_poly).strip('[]'))

########################################################################################################


#############Apply L2 regularization on Ridge Regression using 2rd order polynomial features############
def mnistUsingRidgeRegression():
    alphaSet = [0.001, 0.005,0.01,0.05,0.1,0.5,1]
    accuracySet = []
    for k in alphaSet:
        model = Ridge(alpha=k,normalize = True)
        model.fit(x_train_poly2, y_train_10)
        predicted = model.predict(x_test_poly2)
        # print(predicted[0])
        indexRidge = np.argmax(predicted,axis = 1)
        error = 0
        for i in range(len(indexRidge)):
            if indexRidge[i] != y_test[i]:
                error +=1
        # total No. of errors # 1668
        # print(error)
        accuracy = str(round((1-error/len(indexRidge))*100,2))+'%'
        accuracySet.append(accuracy)
        # print(1-error/len(indexRidge))
    print('The accuracies in Ridge model are: '+str(accuracySet).strip('[]'))
###################################################################################

# MLP classifier
def mnistUsingMLP(train_dataSet, train_labels, test_dataSet, test_labels):

    clf = MLPClassifier(hidden_layer_sizes=(100,),
                        activation='relu', solver='sgd',learning_rate = 'constant',
                        learning_rate_init=0.01, 
                        max_iter=500)
    print('Start to train MLP classifier')
    start = time.time()
    clf.fit(train_dataSet, train_labels)
    print('train completed, time:' + str(time.time() - start))
    print('start to predict and classify:')
    res = clf.predict(test_dataSet) 
    # print(res)
    error_num = 0 
    num = len(test_dataSet) 

    for i in range(num):  
        if res[i] != test_labels[i]:
            error_num += 1
    print("Total num:", num, " Wrong num:",
          error_num, "MLP  CorrectRate:", (1 - error_num / float(num)) * 100, '%')


# KNN classifier
def mnistUsingKNN(train_dataSet, train_labels, test_dataSet, test_labels):
    # neighbors_Set = range(1,20,2)
    neighbors_Set = range(1,6,2)  # due to long executing time, put small list set
    KNN_accuracy_set = []
    for k in neighbors_Set:
        knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=k)
        print('start to train KNN classifier with K value ',k)
        start = time.time()
        knn.fit(train_dataSet, train_labels)
        print('completed in time:' + str(time.time() - start))
        print('start to predict and classify:')
        res = knn.predict(test_dataSet)  
        error_num = 0 
        num = len(test_dataSet)  
        print(num)
        for i in range(num):  
            if res[i] != test_labels[i]:
                error_num += 1
        KNN_accuray = str(round((1 - error_num / float(num))*100,2))+'%'
        print("Total num:", num, " Wrong num:", \
            error_num, "KNN  CorrectRate:", KNN_accuray, "%")
        KNN_accuracy_set.append(KNN_accuray)
    print('The accuracies in KNN are: '+str(KNN_accuracy_set).strip('[]'))
    plt.figure()
    plt.title('Accuracy VS K values in KNN Classifier')
    plt.plot(neighbors_Set,KNN_accuracy_set)
    plt.xlabel('K values')
    plt.ylabel('Accuracy')
    if train_dataSet.shape[1] == 30 :
        plt.savefig('Linear Feature Accuracies in KNN Classifier.png')
    else :
        plt.savefig('Polynomial Feature Accuracies in KNN Classifier.png')

# Logistic Classifier
def mnistUsingLR(train_dataSet, train_labels, test_dataSet, test_labels):

    lr = LogisticRegression(penalty = 'l2',solver='lbfgs',multi_class= 'ovr',max_iter=1000) 
    print('start to train Logistic Regression Classifier')
    start = time.time()
    lr.fit(train_dataSet,train_labels)    
    print('completed in time:' + str(time.time() - start))

    print('start to predict and classify:')
    res = lr.predict(test_dataSet)
    error_num = 0
    num = len(test_dataSet)
    print(num)
    for i in range(num):  
        if res[i] != test_labels[i]:
            error_num += 1
    print("Total num:", num, " Wrong num:", \
          error_num, "Logistic Regression  CorrectRate:", (1-error_num / float(num)) * 100, "%")

# SVM Classifier
def mnistUsingSVM(train_dataSet, train_labels,test_dataSet, test_labels):   
    # C_set = [0.01,0.05,0.1,0.5,1]
    C_set = [0.5,1]  # due to long executing time, put small list set
    SVM_accuracy_set = []
    for c in C_set:
        svc = SVC(C = c,gamma='scale', kernel='rbf')  #kernel='linear'
        print('start to train SVM Classifier using RBF kernel with penalty ',c)
        start = time.time()
        svc.fit(train_dataSet, train_labels)
        print('completed in time:' + str(time.time() - start))
        print('start to predict and classify:')
        res = svc.predict(test_dataSet)
        error_num = 0
        num = len(test_dataSet)

        for i in range(num):
            if res[i] != test_labels[i]:
                error_num += 1
        SVM_accuray = str(round((1 - error_num / float(num))*100,2))+'%'
        print("Total num:", num, " Wrong num:",
            error_num, "SVM  CorrectRate:", SVM_accuray, '%')
        SVM_accuracy_set.append(SVM_accuray)
    print('The accuracies in SVM are: '+str(SVM_accuracy_set).strip('[]'))

if __name__ == '__main__':

    make_print_to_file(path='')
 
    #############################################################
    # From here onwards, all content in print will save to log #
    #############################################################

    ## different eigen vectors on linear and polynomial model
    mnistUsingDiff_eigen_vectors()

    # L2 penalty on Ridge regression 
    mnistUsingRidgeRegression()

    ## linear feature in 30 dimensions
    mnistUsingMLP(x_train, y_train, x_test, y_test)
    mnistUsingKNN(x_train, y_train, x_test, y_test)
    mnistUsingSVM(x_train, y_train, x_test, y_test)
    mnistUsingLR(x_train, y_train, x_test, y_test)
    

    ##polynomial feature on different Classifier
    mnistUsingMLP(x_train_poly2, y_train, x_test_poly2, y_test)
    mnistUsingKNN(x_train_poly2, y_train, x_test_poly2, y_test)
    mnistUsingSVM(x_train_poly2, y_train, x_test_poly2, y_test)
    mnistUsingLR(x_train_poly2, y_train, x_test_poly2, y_test)


 