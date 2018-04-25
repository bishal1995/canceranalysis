from __future__ import division
import numpy as np # linear algebra
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import logging,sys

FORMAT = '%(asctime)-15s [%(levelname)-8s] %(message)s'
logging.basicConfig(stream=sys.stdout,format=FORMAT, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%I')
console = logging.StreamHandler()
console.setLevel(logging.INFO)


#read data from dataset
logging.info("*** LOAD DATASET ***")
dataset = shuffle(np.array(pd.read_csv("dataset.csv",header=1)))

#data frame
logging.info("*** CLEANING DATAFRAME ***")
data_frame = pd.read_csv("dataset.csv",header=1)
data_frame.drop(data_frame.columns[[0]], axis=1, inplace=True)
dataset = shuffle(np.array(data_frame))

extracted_dataset= []
target = []

#extract target column
for row in dataset:
    extracted_dataset.append(row[1:])
    if row[0] == 'B':
        target.append(0)
    else:
        target.append(1)
rffile=open("RandomForestStatistics.txt","w")
acc=0
for i in range(0,100):
	X_train, X_test, Y_train, Y_test= train_test_split(extracted_dataset,target,test_size=0.3)
	logging.info("*** DATASET PARTITIONED IN TRAIN: "+str(len(X_train))+ " TEST: "+str(len(X_test)))
	rf = RandomForestClassifier(n_estimators=18)

	rf = rf.fit(X_train, Y_train)
	predicted = rf.predict(X_test)
	acc_test = metrics.accuracy_score(Y_test, predicted)

	#print ('The accuracy on test data is %s' % (round(acc_test,2)))

	#clf = KNeighborsClassifier(n_neighbors=5,algorithm='brute',p=1)
	#clf.fit(X_train,Y_train)
	logging.info("*** TRAINING END ***")


	#predicted = clf.predict(X_test)
	idx = 0
	true = 0
	false = 0
	for i in X_test:
		#logging.info("*** Pred:"+str(predicted[idx])+" real: "+str(Y_test[idx])+" res "+str(predicted[idx]==Y_test[idx])+" ***")
		if predicted[idx]==Y_test[idx]:
			true +=1
		else:
			false +=1
		idx +=1
	accuracy =  (true/(true+false))*100
	acc=acc+accuracy
	logging.info("Positive Class: "+str(true))
	#rffile.write("\n Positive Class: "+str(true))
	logging.info("Negative Class: "+str(false))
	#rffile.write("\n Negative Class: "+str(false))
	logging.info("Accuracy: "+str(accuracy))
	rffile.write("\n Accuracy: "+str(accuracy))
	cnf_matrix = confusion_matrix(Y_test, predicted)
	print(cnf_matrix)
rffile.write("\nAverage Accuracy: "+str(acc/100))
rffile.write("\n\n")
print("Average Accuracy: ",acc/100)
rffile.close()