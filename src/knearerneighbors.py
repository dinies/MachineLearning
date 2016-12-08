from sklearn import neighbors, datasets, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import colors, pyplot
class knn(object):


	def __init__(self):
		self.do_things()

	def normalize(self, matrix):
		X_normalized = preprocessing.normalize(matrix)
		X_scaled = preprocessing.scale(X_normalized)
		return X_scaled

	def apply_pca(self,matrix, pca_param):
		"return a matrix n x pca_param"
		return PCA(pca_param).fit_transform(matrix)

	def do_things(self):

		iris = datasets.load_iris()
		X=iris.data
		y= iris.target
		X_std=self.normalize(X)
		X_std=self.apply_pca(X_std,2)
		X_train, X_test, y_train, y_test= train_test_split(X_std, y,test_size=0.6)
		clf=neighbors.KNeighborsClassifier(3)
		clf.fit(X_train,y_train)
		score=clf.score(X_test,y_test)
		print(score)
		self.do_plot_decision_boundary(X,clf)

	def do_plot_decision_boundary(self, X,clf):
	#make imports off all these lines !!
		h=1
		x_min, x_max= X[:,0].min() -1, X[:,0].max() +1
		y_min, y_max= X[:,1].min() -1, X[:,1].max() +1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
		Z= clf.predict(np.c_[xx.ravel(), yy.ravel()])
		Z= Z.reshape(xx.shape)
		pyplot.figure()
		cmap_light = colors.ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
		pyplot.pcolormesh(xx, yy, Z, cmap=cmap_light)


x= knn()      
           