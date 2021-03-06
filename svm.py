from __future__ import division
import numpy as np # linear algebra
import pandas as pd
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.model_selection import cross_val_score
import logging,sys

FORMAT = '%(asctime)-15s [%(levelname)-8s] %(message)s'
logging.basicConfig(stream=sys.stdout,format=FORMAT, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%I')
console = logging.StreamHandler()
console.setLevel(logging.INFO)


#read data from dataset
logging.info("*** LOAD DATASET ***")
dataset = shuffle(np.array(pd.read_csv("my_csv_70.csv",header=0)))

#data frame
logging.info("*** CLEANING DATAFRAME ***")
data_frame = pd.read_csv("my_csv_70.csv",header=0)
#data_frame.drop(data_frame.columns[[0]], axis=1, inplace=True)
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


svfile=open("SVMStatistics.txt","w")
acc=0
for i in range(0,5):
	X_train, X_test, Y_train, Y_test= train_test_split(extracted_dataset,target,test_size=0.3)
	logging.info("*** DATASET PARTITIONED IN TRAIN: "+str(len(X_train))+ " TEST: "+str(len(X_test)))

	sm = svm.SVC(kernel='linear', C=1)
	sm.fit(X_train, Y_train)
	#predicted= svc_linear.predict(X_test)
	predicted = sm.predict(X_test)

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
	cnf_matrix = confusion_matrix(Y_test, predicted)

	accuracy =  (true/(true+false))*100
	acc=acc+accuracy
	logging.info("Positive Class: "+str(true))
	logging.info("Negative Class: "+str(false))
	logging.info("Accuracy: "+str(accuracy))
	svfile.write("\nAccuracy: "+str(accuracy))
svfile.write("\nAverage Accuracy"+str(acc/100))
svfile.close();
print(cnf_matrix)