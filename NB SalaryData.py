import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
SalaryTest = pd.read_csv("D:\\ExcelR Data\\Assignments\\Naive bayes\\SalaryData_Test.csv")
SalaryTrain = pd.read_csv("D:\\ExcelR Data\\Assignments\\Naive bayes\\SalaryData_Train.csv")


SalaryTest.columns #to see column names
SalaryTrain.columns
###################### Creating Dummy Variables using "Lable Encoder" ###############################
Le = preprocessing.LabelEncoder()
# SalaryTest
SalaryTest['Workclass']=Le.fit_transform(SalaryTest['workclass'])
SalaryTest['Education'] = Le.fit_transform(SalaryTest['education'])
SalaryTest['Educationno'] = Le.fit_transform(SalaryTest['educationno'])
SalaryTest['Maritalstatus'] = Le.fit_transform(SalaryTest['maritalstatus'])
SalaryTest['Occupation'] = Le.fit_transform(SalaryTest['occupation'])
SalaryTest['Relationship'] = Le.fit_transform(SalaryTest['relationship'])
SalaryTest['Race'] = Le.fit_transform(SalaryTest['race'])
SalaryTest['Sex'] = Le.fit_transform(SalaryTest['sex'])
SalaryTest['Native'] = Le.fit_transform(SalaryTest['native'])
SalaryTest['salary'] = Le.fit_transform(SalaryTest['Salary'])

#SalaryTrain
SalaryTrain['Workclass']=Le.fit_transform(SalaryTrain['workclass'])
SalaryTrain['Education'] = Le.fit_transform(SalaryTrain['education'])
SalaryTrain['Educationno'] = Le.fit_transform(SalaryTrain['educationno'])
SalaryTrain['Maritalstatus'] = Le.fit_transform(SalaryTrain['maritalstatus'])
SalaryTrain['Occupation'] = Le.fit_transform(SalaryTrain['occupation'])
SalaryTrain['Relationship'] = Le.fit_transform(SalaryTrain['relationship'])
SalaryTrain['Race'] = Le.fit_transform(SalaryTrain['race'])
SalaryTrain['Sex'] = Le.fit_transform(SalaryTrain['sex'])
SalaryTrain['Native'] = Le.fit_transform(SalaryTrain['native'])
SalaryTrain['salary'] = Le.fit_transform(SalaryTrain['Salary'])
########### Droping the unwanted columns ###########################
# SalaryTest
SalaryTest.drop(["workclass"],inplace=True,axis=1)
SalaryTest.drop(["education"],inplace=True,axis=1)
SalaryTest.drop(["educationno"],inplace=True,axis=1)
SalaryTest.drop(["maritalstatus"],inplace=True,axis=1)
SalaryTest.drop(["occupation"],inplace=True,axis=1)
SalaryTest.drop(["relationship"],inplace=True,axis=1)
SalaryTest.drop(["race"],inplace=True,axis=1)
SalaryTest.drop(["sex"],inplace=True,axis=1)
SalaryTest.drop(["native"],inplace=True,axis=1)
SalaryTest.drop(["Salary"],inplace=True,axis=1)
#SalaryTrain
SalaryTrain.drop(["workclass"],inplace=True,axis=1)
SalaryTrain.drop(["education"],inplace=True,axis=1)
SalaryTrain.drop(["educationno"],inplace=True,axis=1)
SalaryTrain.drop(["maritalstatus"],inplace=True,axis=1)
SalaryTrain.drop(["occupation"],inplace=True,axis=1)
SalaryTrain.drop(["relationship"],inplace=True,axis=1)
SalaryTrain.drop(["race"],inplace=True,axis=1)
SalaryTrain.drop(["sex"],inplace=True,axis=1)
SalaryTrain.drop(["native"],inplace=True,axis=1)
SalaryTrain.drop(["Salary"],inplace=True,axis=1)


# Splitting data into xtrain and xtest
xtest= SalaryTest.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]]
ytest= SalaryTest.iloc[:,[13]]
xtrain= SalaryTrain.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]]
ytrain= SalaryTrain.iloc[:,[13]]
##################### With GaussianNB ##################################
gnb = GaussianNB() # normal distribution


# Building and predicting at the same time 

pred_gnb = gnb.fit(xtrain,ytrain).predict(xtest)



# Confusion matrix GaussianNB model
confusion_matrix(ytest,pred_gnb) # GaussianNB model
pd.crosstab(ytest.values.flatten(),pred_gnb) # confusion matrix using 
np.mean(pred_gnb==ytest.values.flatten()) # 0.7946879150066402
#np.mean(pred_gnb==ytest.values.flatten())*100 To get Answer in % = 79.46879150066401

#################### With MultinomialNB ############################
# This is most the time used for Textule data
mnb = MultinomialNB()

# Building and predicting at the same time 

pred_gnb = mnb.fit(xtrain,ytrain).predict(xtest)

# Confusion matrix MultinomialNBmodel
confusion_matrix(ytest,pred_gnb) # MultinomialNB model
pd.crosstab(ytest.values.flatten(),pred_gnb) # confusion matrix using 
np.mean(pred_gnb==ytest.values.flatten()) #0.7749667994687915
#np.mean(pred_gnb==ytest.values.flatten())*100 To get Answer in % = 77.49667994687915