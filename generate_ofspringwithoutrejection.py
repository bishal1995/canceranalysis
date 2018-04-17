import pandas as pd
import numpy as np
import random
from tqdm import tnrange 

# Reading data
cancer = pd.read_csv('dataset.csv').drop(['id','Unnamed: 32'],axis=1).fillna('0')
malignant = cancer[cancer.diagnosis=='M'].drop(['diagnosis'],axis=1)
benign = cancer[cancer.diagnosis == "B"].drop(['diagnosis'],axis=1)
mal_count = len(malignant)
ben_count = len(benign)

# Function for returning random number sequence
def rand_sequence():
    no_exchange_parameter = 15 #
    no_parameter = 30
    return np.random.randint(1,no_parameter,no_exchange_parameter)

# Getting some values
total = mal_count*ben_count
mean_m = malignant.mean().values
mean_b = benign.mean().values

# Function for choosing whether given data is malignamt or benign
def chose_mb(mean_m,mean_b,row):
    val = abs(row - mean_m) > abs(row - mean_b)
    no_params = 30
    m_count = val.sum()
    percent = (m_count*100)/no_params
    if( percent > 70.00 ):
        return 'M'
    else:
        return 'B'
# Generating offspring
total = mal_count * ben_count * 2
mcount = 0
bcount = 0
for m in range(mal_count):
    df = pd.DataFrame(columns=['Diagnosis']+malignant.columns.tolist(),index=range(total))
    row_count=0
    for b in range(ben_count):
        rand_idx = rand_sequence()
        temp_m = malignant.iloc[m].values
        temp_b = benign.iloc[b].values
        for k in rand_idx:
            temp_b[k],temp_m[k]=temp_m[k],temp_b[k]
        diag1 = chose_mb(mean_m,mean_b,temp_b)
        diag2 = chose_mb(mean_m,mean_b,temp_m)
        if( diag1 == 'M' ):
            mcount+=1
        else:
            bcount+=1
        if( diag2 == 'M' ):
            mcount+=1
        else:
            bcount+=1
        df.iloc[row_count]=[diag1] + temp_b.tolist()
        row_count+=1
        df.iloc[row_count]=[diag2] + temp_m.tolist()
        row_count+=1
    df.to_csv('my_csv1_70.csv', mode='a', header=False,index=False)
    mal_percent = (mcount*100)/((m+1)*(b+1)*2)
    ben_percent = (bcount*100)/((m+1)*(b+1)*2)
    print(
        ' Done : ' + str( ((m+1)*(b+1)*100)/total ) + '%' +
        ' Benign : ' + str( ben_percent )  + '%' +
        ' Malignant : ' + str( mal_percent ) + '%'
        ,end='\r'
    )
