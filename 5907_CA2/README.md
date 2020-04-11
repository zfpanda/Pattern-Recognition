# EE5907 CA2 Pattern Recognition


### Prerequisites

This assignment is coded in Python with version 3.7.4. 
And modules like matplotlib, numpy, scipy are imported. Make sure all the modules 
are installed in the correct environment before execute the Python code.

### Mnist Database file
pls copy the 4 zip files to the src folder before running all the scripts
since gz. files are not included by requirement.

### Run

Open terminal, cd into the corresponding folder path under `src`. 
````you maybe need to remove the [] in the foldername in order reach the path````

For example, for Qb:
````
cd src/Qb
````
 and run `Qb.py` by typing: 
````
python Qb.py
````
Then press `Enter` to execute.

### For Qb
In the Qb folder, when b.scripy successfully executed, eigen vectors, first 10 row images from tran set, 
and second row digits reconstrcuted from test will be saved in the current directory path.


### For Qc1
Qc1.py is the results obtained by psudoinverse matrix method
ROC and Confusion Matrix will be saved

Qc1SGD.py is the results obtained by Stochastic Gradient Descent (SGD) method:
ROC and Confusion Matrix will be saved

### For Qc2
Qc2.py is the results obtained by psudoinverse matrix method
ROC and Confusion Matrix will be saved

Qc2SGD.py is the results obtained by Stochastic Gradient Descent (SGD) method:
ROC and Confusion Matrix will be saved

### For Qd:
In the Qd folder, the script contains functions as such:
	## different eigen vectors on linear and polynomial regression models
    mnistUsingDiff_eigen_vectors()

    # L2 penalty on Ridge regression 
    mnistUsingRidgeRegression()

    ##PCA linear feature of 30 dimensions on different Classifier
    mnistUsingMLP(x_train, y_train, x_test, y_test)
    mnistUsingKNN(x_train, y_train, x_test, y_test)
    mnistUsingSVM(x_train, y_train, x_test, y_test)
    mnistUsingLR(x_train, y_train, x_test, y_test)
    

    ##polynomial feature on different Classifier
    mnistUsingMLP(x_train_poly2, y_train, x_test_poly2, y_test)
    mnistUsingKNN(x_train_poly2, y_train, x_test_poly2, y_test)
    mnistUsingSVM(x_train_poly2, y_train, x_test_poly2, y_test)
    mnistUsingLR(x_train_poly2, y_train, x_test_poly2, y_test)

### You can test anyone of the classifer or function by comment out rest of classifier.
##the polynomial regression may run out of Ram memory, make sure the machine has enough memory size. 
## a log function in the Qd will write all the print content to the log file once the script successfully executed 
## to run all the functions at one time in Qd may take some time. you can selectively run and test any function accordingly. 



### Author
Zhang Fei - A0117981X

##Thank you!