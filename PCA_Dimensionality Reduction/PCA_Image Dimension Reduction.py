from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import sys

#Accept the grayscale image as an input and assign the pixel values to a numpy array
im = Image.open("test.jpg")
A = np.array(im)

#Height and width of the image in pixels
m , n = A.shape

#Display the test image loaded into the array
print("The original input image in {}x{} pixel format:".format(m,n))
plt.imshow(A, cmap='gray')
plt.show()

#User defines how many features to retain as part of Principal Component Analysis
print("\nEnter reduced number of features for PCA reduction (Original feature size: {}):".format(n))
no_of_pca = int(input())


if(no_of_pca<=0 or no_of_pca>n):
    print("Reduced dimension cannot be more than original dimension or less than 1.")
    sys.exit(1)
    

#Vector M stores the mean values of each column in the pixel matrix A
M = np.mean(A, axis=0)

#Calculate covariance matrix
C = np.dot(np.transpose((A-M)),(A-M))/(len(A)-1) #Mean normalization is done before calculation of covariance matrix

#Calculate eigen vectors (direction of each dimension) and eigen values (amount of changes/variance in the same direction)
eigen_vals, eigen_vecs = np.linalg.eig(C)

#eigen values and eigen vectors are rounded off to float values to remove the imaginary (complex) part if any
eigen_vals = eigen_vals.astype('float64')
eigen_vecs = eigen_vecs.astype('float64')

#Add the eigen values and the corresponding eigen vectors into a list of tuples
eig_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

#Sort the list of tuples in descending order of eigen values- so that the eigen vectors at the top denote the dimensions having maximum data variance
eig_pairs.sort(key=lambda x: x[0], reverse=True)

#As part of PCA, we will take only no_of_pca count of eigen vectors to perform dimensionality reduction
eigen_vecs_reduced = []
for i in range(no_of_pca):
    temp_tup = eig_pairs[i]
    temp_list = temp_tup[1]
    eigen_vecs_reduced.append(temp_list)

#Covert the list to a numpy array
eigen_vecs_reduced = np.array(eigen_vecs_reduced)
eigen_vecs_reduced = np.transpose(eigen_vecs_reduced)

#Dimensionality reduction using PCA principle (image compression) - keeping only no_of_pca count of features per row
A_lower = np.dot(A, eigen_vecs_reduced)

#Transforming back the image to the oridinal dimension (mxn) 
A_approx = np.dot(A_lower,np.transpose(eigen_vecs_reduced))+M #Adding back the mean values initially deducted to perform mean normalization

#Render the image
plt.imshow(A_approx, cmap='gray')
plt.show()

#Save the image
matplotlib.image.imsave('test_Post PCA Reduction.jpg', A_approx, cmap='gray')