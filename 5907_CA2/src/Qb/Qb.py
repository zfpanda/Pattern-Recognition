import numpy as np
import struct
import matplotlib.pyplot as plt
import gzip

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

# subtract the mean from each training data point
x_train = x_train_raw - meanVal

# Compute the covariance matrix using all the training images
covMat = np.cov(x_train,rowvar = 0)

# get eigenvalue and eigenvector from covirance matrix
featValue,featVec = np.linalg.eig(covMat)
# sort the eigenvalue in ascending order and return index
index = np.argsort(featValue)

n_index=index[-1:-11:-1] #get indexes of the 10 largest eigenvalues
n_featVec=featVec[:, n_index] # get eigenvectors matrix corresponding to the largest k dimension eigenvalues

# display First 10 eigen-digits from train set
for i in range(10):
    plt.subplot(2,5,i+1)
    img = n_featVec[:,i].reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.savefig('First 10 eigen-digits.png')
plt.show()
plt.clf()

# Handwritten digit image reconstruction
# in python index starts from 0 which is different from matlab which index starts from 1. all index minus 1
# first_row_idx = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# second_row_idx = [4, 3, 2, 19, 5, 9, 12, 1, 62, 8]
first_row_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
second_row_idx = [3, 2, 1, 18, 4, 8, 11, 0, 61, 7]
first_row_img = x_train_raw[first_row_idx]
second_row_img = x_test_raw[second_row_idx]

n_30_index=index[-1:-31:-1] #get indexes of the 30 largest eigenvalues
n_30_featVec=featVec[:, n_30_index] # get eigenvectors matrix corresponding to the largest k dimension eigenvalues

# Reconstruct images from test set
lowData=np.dot(second_row_img,n_30_featVec) #lowData=second_row_img*n_30_featVec
highData=np.dot(lowData,n_30_featVec.T) #highData=(lowData*n_30_featVec.T)

# Display first row images from train set
for i in range(10):
    plt.subplot(2,5,i+1)
    img = first_row_img[i].reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.savefig('First row images from train set.png')
plt.show()
plt.clf()

# Display Second row reconstructed images from test set
for i in range(10):
    plt.subplot(2,5,i+1)
    img = highData[i].reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.savefig('Second row reconstructed images from test set.png')
plt.show()
plt.clf()