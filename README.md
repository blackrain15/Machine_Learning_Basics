# Machine Learning : A Journey towards an Automated and Connected World

The project intends to implement some of the widely used regression and classification algorithms from scratch using Python. This is a platform of collaboration to fine-tune the algorithms even further to optimize performance and accuracy. The code that I have developed can be leveraged across different use cases of Machine learning with necessary customization. This is contribution to the open source community in the space of Machine Learning to learn and grow together and aim for a future driven by AI and Machine learning. <BR>

Some of the implementations checked in the repository can be referred from the below links:

### **Linear Regression - Solved with Normal Equation :** <BR>

The implementation takes into consideration a small data-set of ~30 records, with 2 variables (x and y) available in each of the records, generated synthetically in the run-time. Normal equation is applied on the data-set to derive the intercept and coefficient of x and finally, the linear regression line is derived to predict the value of y for a given value of x. Matplotlib library is used to plot the initial data-set as well as the decision boundary as derived by Normal Regression. <BR>

* [Code-base](https://github.com/blackrain15/Machine_Learning_Basics/tree/master/Linear%20Regression_Normal%20Equation-:-Code-base)

### **Linear Regression - Solved with Gradient Descent :** <BR>

The implementation takes into consideration a training data-set of ~100 records available in a text file. Each of the records in the training data-set has one feature variable and one output variable. The values of intercept and coefficient of x is initialized with random values. The hyper-parameter alpha is assigned with an appropriate value as decided from multiple runs. An iterative approach is followed to compute the costs and incrementally fine-tune the values of intercept and coefficient. When the change in costs across multiple iterations falls below a pre-defined threshold limit, the iteration stops and the values of coefficients are treated as optimal. With the coefficients derived using Gradient Descent algorithm, the values of y are predicted against the given values of x. Matplotlib library is used to plot the initial data-set as well as the predicted values. The cost function is also plotted against the number of iterations to ensure that the cost function gradually converges and Gradient Descent is successful. <BR>

* [Code-base](https://github.com/blackrain15/Machine_Learning_Basics/tree/master/Linear%20Regression_Gradient%20Descent-:-Code-base)

### **Multi-variate Linear Regression :** <BR>

A training data-set of ~100 records, available in a text file, is used as a training dataset. Each of the records in the training data-set has 2 feature variables (area of flat, number of rooms) and one output variable (cost). The values of intercept (x0) and coefficients of x (x1, x2) are initialized with random values. The hyper-parameter alpha is assigned with an appropriate value as decided from multiple runs. An iterative approach is followed to compute the costs and incrementally fine-tune the values of intercept and coefficient. When the change in costs across multiple iterations falls below a pre-defined threshold limit, the iteration stops and the values of coefficients are treated as optimal. With the coefficients derived using Gradient Descent algorithm, the values of y (i.e. Costs) are predicted against the given values of x (Area of the flat, Number of rooms). Matplotlib library is used to plot the initial data-set as well as the predicted values. The cost function is also plotted against the number of iterations to ensure that the cost function gradually converges and Gradient Descent is successful. <BR>

* [Code-base](https://github.com/blackrain15/Machine_Learning_Basics/tree/master/Multi-variate%20Linear%20Regression_Gradient%20Descent-:-Code-base)

### **Logistic Regression :** <BR>

This is a Python based implementation of Logistic Regression which is a classification algorithm. The training data-set is referred from a text file - having 3 variables against each of the training records. Each of the records in the training data-set has 2 feature variables (mark in Physics and mark in Mathematics) and one output variable which denotes whether the student is selected for the university (1) or Not Selected (0). The values of intercept (x0) and coefficients of x (x1, x2) are initialized with random values. The hyper-parameter alpha is assigned with an appropriate value as decided from multiple runs. An iterative approach is followed to compute the costs and incrementally fine-tune the values of intercept and coefficient. When the change in costs across multiple iterations falls below a pre-defined threshold limit, the iteration stops and the values of coefficients are treated as optimal. With the coefficients derived using Gradient Descent algorithm, the values of y (i.e. Costs) are predicted against the given values of x (Area of the flat, Number of rooms). Matplotlib library is used to plot the initial data-set as well as the predicted values. The cost function is also plotted against the number of iterations to ensure that the cost function gradually converges and Gradient Descent is successful. <BR>

* [Code-base](https://github.com/blackrain15/Machine_Learning_Basics/tree/master/Logistic%20Regression-:-Code-base)

### **K-Means Clustering :** <BR>

The objective is to come up with a Python based implementation of K-Means Clustering algorithm - which is a unsupervised clustering algorithm. The user can decide the training set size (m) and generate a (mx2) matrix with randomized values as training set. The user can also enter the number of clusters (k) that is needed to be derived by the K-Means Clustering algorithm. The cluster centroids are assigned as randomly selected k data-points selected from the training data-set. An iterative approach is followed to derive the distances of each of the training data-points from the different cluster-centroids. Each data-point is assigned to the cluster having the nearest cluster-centroid with respect to the data-point. Total cost is derived against each iteration considering the sum of squared errors - i.e. the squared value of norm of the data-points with respect to corresponding cluster-centroids. As part of the iterations, the cluster-centroids are incrementally changed to the mean of the cluster data-points. Iterative approach continues until the cluster-centroids stop shifting and the cost function saturates. Matplotlib library is used to plot the initial data-set as well as the clusters getting changed after every iteration. The code snippet can be further customized to be run iteratively with different initial assignments of cluster-centroids and finally, the cluster selection having minimum variance can be chosen as the optimal sets. <BR>

* [Code-base](https://github.com/blackrain15/Machine_Learning_Basics/tree/master/K-Means%20Clustering-:-Code-base)
