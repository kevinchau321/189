import scipy.io as sio
import numpy as np
from random import randint
from sklearn import svm, datasets
from sklearn import utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

def trainSVM(n = 100, X = np.zeros([60000,784]), y = np.zeros([60000,1])):
	clf = SVC(C = 1.0)
	clf.fit(X[:n],y[:n])
	return clf

def errorRate(n = 100, C = 1.0):
	trainingdata = sio.loadmat('spam_data.mat')
	X = trainingdata['training_data']
	y = np.array(trainingdata['training_labels']).transpose()
	numsamples = X.shape[0]
	X, y = utils.shuffle(X,y)
	clf = LinearSVC(C = C)
	Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, train_size=n/(1.0*numsamples))
	clf.fit(Xtrain, ytrain)
	wrong = 0
	Z = clf.predict(Xtest)
	for i in range(0,numsamples-n):
		if Z[i] != ytest[i]:
			wrong = wrong + 1

	print('error is: ')
	print(wrong/10000.0)
	return wrong/10000.0

def plotRateVsExamples():
	x = [100,200,500,1000,2000,5000,10000]
	y = [0,0,0,0,0,0,0]
	i = 0
	for num in x:
		y[i] = errorRate(num)
		i = i + 1
	plt.plot(x, y, 'ro')
	plt.axis([0, 10000, 0, 1])
	plt.savefig('p1.png')
	plt.show()
	return

def makeConfusionMatrix(n=100):
	# import some data to play with
	trainingdata = sio.loadmat('spam_data.mat')
	X = np.swapaxes(trainingdata['training_data'].reshape(784,60000), 0, 1)
	y = np.array(trainingdata['training_labels']).transpose()[0]
	# Split the data into a training set and a test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n/60000.0)

	# Run classifier
	classifier = svm.SVC(kernel='linear')
	y_pred = classifier.fit(X_train, y_train).predict(X_test)
	# Compute confusion matrix
	cm = confusion_matrix(y_test, y_pred)

	print(cm)

	# Show confusion matrix in a separate window
	plt.matshow(cm)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig('matrix'+str(n))
	plt.show()
	return

def makeConfusionMatrices():
	for n in [100,200,500,1000,2000,5000,10000]:
		makeConfusionMatrix(n)

def crossValidation10fold():
	trainingdata = sio.loadmat('spam_data.mat')
	X = trainingdata['training_data']
	y = np.array(trainingdata['training_labels']).transpose()
	X, y = utils.shuffle(X,y)
	#Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, train_size=n/60000.0)
	numsamples = X.shape[0]
	leastCVerror = 1
	bestC = 0
	for c in [.1,.2,.5,1,2,5,10,15,18,19,20,21,21.5,22.05,22.5,23,23.5,24,25,50,100,200, 500, 1000]:
		print('C is now: '+str(c))
		clf = LinearSVC(C = c)
		total = 0
		CVerror = 0

		for i in range(0,10):
			Xtest = X[i*(numsamples/10):i*(numsamples/10)+(numsamples/10)]
			Xtrain = np.append(X[:i*(numsamples/10)],X[i*(numsamples/10)+(numsamples/10):numsamples],axis=0)
			ytest = y[i*(numsamples/10):i*(numsamples/10)+(numsamples/10)]
			ytrain = np.append(y[:i*(numsamples/10)],y[i*(numsamples/10)+(numsamples/10):numsamples])
			clf.fit(Xtrain, ytrain)
			wrong = 0
			Z = clf.predict(Xtest)
			for i in range(0,(numsamples/10)):
				if Z[i] != ytest[i]:
					wrong = wrong + 1
			error = wrong/(1.0*(numsamples/10))
			print(error)
			total += error
		CVerror = total/10.0
		print('cross validation error is: '+ str(CVerror))
		if CVerror < leastCVerror:
			leastCVerror = CVerror
			bestC = c
	print('optimal error is: '+str(leastCVerror))
	print('optimal C is: '+str(bestC))
	validationerror = errorRate(n=1000,C=bestC)
	return validationerror

def testKaggle():
	predictionfile = open('spam.csv','w+')
	predictionfile.write('Id,Category\n')
	trainingdata = sio.loadmat('spam_data.mat')
	X = trainingdata['training_data']
	y = np.array(trainingdata['training_labels']).transpose()
	X, y = utils.shuffle(X,y)
	clf = LinearSVC(C = 50)
	clf.fit(X, y)
	wrong = 0
	Xtest = trainingdata['test_data']
	Z = clf.predict(Xtest)
	for i in range(0,trainingdata['test_data'].shape[0]):
		predictionfile.write(str(i+1)+','+str(Z[i])+'\n')
	return