clf = svm.SVC(gamma=0.0000001,kernel='rbf',C=1000,verbose=True,max_iter=100,decision_function_shape='ovo')

clf.fit(X_train,y_train)



pk_out = open("SVM_gamma=0.0000001,kernel='rbf',C=1000.pickle","rb")
clf = pickle.load(pk_out)
img=plt.imread('1_0_0_20161219140627985.jpg.chip.jpg') 
scale = 100
dim = (scale, scale)
img = cv2.resize(img, dim)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img.flatten()
array = [img]
print
age=clf.predict(array)
print(age)
pk_out.close()
