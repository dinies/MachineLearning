import numpy as np
from sklearn import preprocessing
vec1= [2,1]
vec2= [3,2]
vec3= [1,3]
vec4= [5,4]
vec5= [4,5]



mat= np.matrix([vec1,vec2,vec3,vec4,vec5])
print(mat.shape)
print(mat)
print("standardized")

mat= preprocessing.normalize(mat)
mat= preprocessing.scale(mat)

print(mat.mean())
print(mat.std())


v1=[1,3,6,8]
v2=[2,4,5,7]
v3=[9,10,11,12]
m= np.matrix([v1,v2,v3])
print(m)
print(m[:,3])
#m= normalize(m)


#print ("seconf")
#print(m)