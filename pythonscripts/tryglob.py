from PIL import Image
import numpy as np
import glob
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#find paths of 4 object classes and store them in a dictionary string->list
objects= ["19","23","76","100"]
imagefolderpath='/home/dinies/Pictures/coil-100/obj'
paths_obj= {}
images= []
for obj in objects:
	paths_obj=glob.glob(imagefolderpath+obj+'_*')
	for path in paths_obj:
				img= np.asarray(Image.open(path))
				imgraveled= img.ravel()
				images.append(imgraveled)
nsamples= len(images)
print(nsamples)
y=np.arange(nsamples)

#images contains [image1,image2,image3] that are vectors of dim (49152,) 
#now i have to find a way to create a matrix from a list of vectors
matrix= np.matrix(images)

print(matrix.shape)