from PIL import Image
import numpy as np
import glob
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
from sklearn.naive_bayes import GaussianNB

#params
objects= ["12","10"]
PCAparam=10
principalcomponent_1=0
principalcomponent_2=1

#crafting matrix from image vectors
#path for linuxsystems imagefolderpath='/home/dinies/Pictures/coil-100/obj'
imagefolderpath=r'C:\Users\user\Pictures\coil\obj'

paths_obj= {}
images= []
y=[]
for obj in objects:
	paths_obj=glob.glob(imagefolderpath+obj+'_*')
	for path in paths_obj:
				img= np.asarray(Image.open(path))
				imgraveled= img.ravel()
				images.append(imgraveled)
				y.append(int(obj))
matrix= np.matrix(images)
#normalizing & PCAing            #print(matrix.shape) : print (matrix.mean())
normalizedmatrix = preprocessing.normalize(matrix)
scaledmatrix = preprocessing.scale(normalizedmatrix)
#print (scaledmatrix.mean()): print (scaledmatrix.std())
X_t = PCA(PCAparam).fit_transform(scaledmatrix)
#print (X_t.shape)

#colors
def generate_color():
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))
    return color

#scattering classes    
incrementer=0
for label in objects:
	n_samples= y.count(int(label))
	rows=range(incrementer,incrementer+n_samples-1)
	k=generate_color()
	plt.scatter(X_t[rows, principalcomponent_1], X_t[rows, principalcomponent_2],c= [k,])
	incrementer= incrementer + n_samples
	
#plt.show()

clf=GaussianNB()
clf.fit(X_t,y)

#TEST WITH ONE IMAGE
label_test= "51_*"
#test_image= np.asarray(Image.open(imagefolderpath+label_test))
#test_image= test_image.ravel()
#list=[test_image]
#test_image=np.matrix(list)
#test_image= preprocessing.normalize(test_image)
#test_image=preprocessing.scale(test_image)
#M= PCA(PCAparam).fit_transform(test_image)
#print("PREDICT")
#print(clf.predict(M))

#TEST WITH A CLUSTER

paths_obj= {}
images= []
y=[]
for obj in objects:
	paths_obj=glob.glob(imagefolderpath+label_test)
	for path in paths_obj:
				img= np.asarray(Image.open(path))
				imgraveled= img.ravel()
				images.append(imgraveled)
				y.append(int(obj))
matrix= np.matrix(images)
#normalizing & PCAing            #print(matrix.shape) : print (matrix.mean())
normalizedmatrix = preprocessing.normalize(matrix)
scaledmatrix = preprocessing.scale(normalizedmatrix)
#print (scaledmatrix.mean()): print (scaledmatrix.std())
X_test = PCA(PCAparam).fit_transform(scaledmatrix)
print("PREDICT CKUSTER")
print(clf.predict(X_test))
