import numpy as np
from PIL import Image
import glob
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import random
from sklearn.naive_bayes import GaussianNB
class bysclf(object):

	def __init__(self,objects=["1","2","3"],pca_param=2,first_comp=0,second_comp=1):
		self.chosen_classes= objects
		self.pca_param=pca_param
		self.first_comp=first_comp
		self.second_comp=second_comp
		self.num_classes= len(objects)
		self.images_path='/Users/edoardoghini/Pictures/coil-100/obj';
		


	def load_in_matrix(self,objects):
		"return a list with a  matrix n x features and an array of labels"
		paths_obj= {}
		images= []
		y=[]
		for obj in objects:
			paths_obj=glob.glob(self.images_path+obj+'_*')
			for path in paths_obj:
				img= np.asarray(Image.open(path))
				imgraveled= img.ravel()
				images.append(imgraveled)
				y.append(int(obj))
		X_raw= np.matrix(images)
		X_normalized = preprocessing.normalize(X_raw)
		X_scaled = preprocessing.scale(X_normalized)
		return [X_scaled,y]

	def generate_color(self):
		"return a rgba format random color"
		color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))
		return color
	def pc_analyze(self,matrix):
		"return a matrix n x pca_param"
		return PCA(self.pca_param).fit_transform(matrix)

	def scatter_plot(self,matrix,label_vector):
		"return void"
		incrementer=0
		for label in self.chosen_classes:
			num_samples=self.find_num_samples(label,label_vector)
			rows=range(incrementer,incrementer+num_samples-1)
			k=self.generate_color()
			plt.scatter(matrix[rows, self.first_comp],matrix[rows, self.second_comp],c= [k,])
			incrementer= incrementer + num_samples
		plt.show()

	def training_classifier(self, classifier, matrix, features_vector):
		classifier.fit(matrix,features_vector)

	def testing_classifier(self, classifier, testing_matrix, testing_label, flag_predict=False):
		if flag_predict :
			return classifier.predict(testing_matrix)
		else:
			return classifier.score(testing_matrix, testing_label)
		
	def do_plot(self):
		[X,y]=self.load_in_matrix(self.chosen_classes)
		X_t=self.pc_analyze(X)
		self.scatter_plot(X_t,y)

	def do_extimation(self ,mimic_labels ,flag_predict=False):
		"Return the score of an extimation of new similar classes @param mimic_label list of strings"
		[X,y]=self.load_in_matrix(self.chosen_classes)
		X_t=self.pc_analyze(X)
		clf=GaussianNB()
		self.training_classifier(clf,X_t,y)
		[X_mimic, y_mimic]= self.load_in_matrix(mimic_labels)
		X_t_mimic=self.pc_analyze(X_mimic)		
		return self.testing_classifier(clf,X_t_mimic,y_mimic,flag_predict)
		
	def do_split_validation(self, rate=0.6):
		[X,y]=self.load_in_matrix(self.chosen_classes)
		X_t=self.pc_analyze(X)
		clf=GaussianNB()
		X_train, X_test, y_train, y_test= train_test_split(X_t, y,test_size=rate) # [,random_state=1]
		self.training_classifier(clf,X_train,y_train)
		return self.testing_classifier(clf,X_test,y_test)

	def do_compare_split_efficence(self):
		vec=np.arange(4,7)
		for i in vec: 
			score= self.do_split_validation(float(i/10))
			plt.scatter(i,score)
		plt.show()


	def find_num_samples(self,label,label_vector):
		return label_vector.count(int(label))
