# %% [code] {"id":"li0O51557j5f","execution":{"iopub.status.busy":"2022-06-24T04:17:30.277075Z","iopub.execute_input":"2022-06-24T04:17:30.277777Z","iopub.status.idle":"2022-06-24T04:17:31.704841Z","shell.execute_reply.started":"2022-06-24T04:17:30.277682Z","shell.execute_reply":"2022-06-24T04:17:31.703569Z"}}
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import  pickle

# %% [markdown] {"id":"GNGfE94c8hE6"}
# Data Collection and Analysis

# %% [code] {"id":"5Q2FJFvL8maV","outputId":"f1d3f510-cc8c-469d-f7bd-8e28ccc18dcc","execution":{"iopub.status.busy":"2022-06-24T04:17:31.706933Z","iopub.execute_input":"2022-06-24T04:17:31.7073Z","iopub.status.idle":"2022-06-24T04:17:31.731516Z","shell.execute_reply.started":"2022-06-24T04:17:31.70727Z","shell.execute_reply":"2022-06-24T04:17:31.730254Z"}}
#loading the data from csv file to pandas dataFrame
parkinsons_data=pd.read_csv('./archive_voice/parkinsons.data')

# %% [code] {"id":"hzuHLHAwArh3","execution":{"iopub.status.busy":"2022-06-24T04:17:31.737257Z","iopub.execute_input":"2022-06-24T04:17:31.740273Z","iopub.status.idle":"2022-06-24T04:17:31.78653Z","shell.execute_reply.started":"2022-06-24T04:17:31.74021Z","shell.execute_reply":"2022-06-24T04:17:31.78508Z"}}
#to print first 5 rows
parkinsons_data.head()

# %% [code] {"id":"z3bMnENkA1O-","outputId":"1b2fa726-1516-4900-fa52-4073422eed6a","execution":{"iopub.status.busy":"2022-06-24T04:17:31.789878Z","iopub.execute_input":"2022-06-24T04:17:31.790368Z","iopub.status.idle":"2022-06-24T04:17:31.797633Z","shell.execute_reply.started":"2022-06-24T04:17:31.790326Z","shell.execute_reply":"2022-06-24T04:17:31.796758Z"}}
#no of rows and columns present in dataset(195 rows and 25 columns)
parkinsons_data.shape

# %% [code] {"id":"TEp4OXXBBQ6X","outputId":"ce8f1fc4-33c7-4845-d401-c971559c3035","execution":{"iopub.status.busy":"2022-06-24T04:17:31.798876Z","iopub.execute_input":"2022-06-24T04:17:31.799821Z","iopub.status.idle":"2022-06-24T04:17:31.828788Z","shell.execute_reply.started":"2022-06-24T04:17:31.799783Z","shell.execute_reply":"2022-06-24T04:17:31.827786Z"}}
#getting more informatio about dataset
parkinsons_data.info

# %% [code] {"id":"D3yu5ikIBYY2","outputId":"a8e74722-714b-4b8c-a587-5996de125f97","execution":{"iopub.status.busy":"2022-06-24T04:17:31.830301Z","iopub.execute_input":"2022-06-24T04:17:31.830986Z","iopub.status.idle":"2022-06-24T04:17:31.841279Z","shell.execute_reply.started":"2022-06-24T04:17:31.830947Z","shell.execute_reply":"2022-06-24T04:17:31.839991Z"}}
#checking for missing values in each column
#if we have missing values we need to handle it before feeding to ml model
parkinsons_data.isnull().sum()

# %% [code] {"id":"pEBRpoNRCNIi","outputId":"b1043450-fdd2-47d7-afec-e4a51a5b7e9b","execution":{"iopub.status.busy":"2022-06-24T04:17:31.844318Z","iopub.execute_input":"2022-06-24T04:17:31.845215Z","iopub.status.idle":"2022-06-24T04:17:31.960163Z","shell.execute_reply.started":"2022-06-24T04:17:31.845161Z","shell.execute_reply":"2022-06-24T04:17:31.958665Z"}}
#getting some statistical measures about the data-mean,avg,count(gives no of values in each column)
parkinsons_data.describe()

# %% [code] {"id":"syIK9rKsCsgu","outputId":"d32844fb-1fbc-4d41-b32f-93b52643d07b","execution":{"iopub.status.busy":"2022-06-24T04:17:31.961547Z","iopub.execute_input":"2022-06-24T04:17:31.961872Z","iopub.status.idle":"2022-06-24T04:17:31.972213Z","shell.execute_reply.started":"2022-06-24T04:17:31.961846Z","shell.execute_reply":"2022-06-24T04:17:31.970934Z"}}
#check how many ppl have parkinson and how many dont have
# this ca be done by checking the distribution of target variable(in this case status column is target column)
parkinsons_data["status"].value_counts()
#in this case 147 ppl have disease(bcz 1)
#48 ppk dont have disease

# %% [code] {"id":"LDBfO67_EqF1"}


# %% [code] {"id":"kQWpkWJjDCd8","outputId":"47e673b3-4f01-4590-d27b-f92f154636bf","execution":{"iopub.status.busy":"2022-06-24T04:17:31.974188Z","iopub.execute_input":"2022-06-24T04:17:31.975573Z","iopub.status.idle":"2022-06-24T04:17:32.007092Z","shell.execute_reply.started":"2022-06-24T04:17:31.975508Z","shell.execute_reply":"2022-06-24T04:17:32.005878Z"}}
# take mean of all values of ppl who have parkinson and who dont have parkinson
parkinsons_data.groupby('status').mean()

# %% [markdown] {"id":"FiqnrGNNE_2f"}
# Data Preprocessing

# %% [code] {"id":"k61LXYb8FEKO","execution":{"iopub.status.busy":"2022-06-24T04:17:32.012246Z","iopub.execute_input":"2022-06-24T04:17:32.012967Z","iopub.status.idle":"2022-06-24T04:17:32.020928Z","shell.execute_reply.started":"2022-06-24T04:17:32.012918Z","shell.execute_reply":"2022-06-24T04:17:32.019675Z"}}
#seperate features(all other column) and target(status column)
#droping column axis=1
#droping row -axis=0
x=parkinsons_data.drop(columns=['name','status'],axis=0)
y=parkinsons_data['status']

# %% [code] {"id":"UC7vOc_TFx7u","outputId":"fa3d43b8-d52d-4c5d-e1a1-48e83809b37b","execution":{"iopub.status.busy":"2022-06-24T04:17:32.023608Z","iopub.execute_input":"2022-06-24T04:17:32.024693Z","iopub.status.idle":"2022-06-24T04:17:32.056332Z","shell.execute_reply.started":"2022-06-24T04:17:32.024635Z","shell.execute_reply":"2022-06-24T04:17:32.054982Z"}}
print(x)
print(y)

# %% [markdown] {"id":"pr3g6KbFF3kI"}
# Spliting dataset into training and testing 
# 

# %% [code] {"id":"DomLh_dcGN1x","outputId":"60a47114-087a-49a2-9025-ed56823ad3dc","execution":{"iopub.status.busy":"2022-06-24T04:17:32.058533Z","iopub.execute_input":"2022-06-24T04:17:32.062187Z","iopub.status.idle":"2022-06-24T04:17:32.080279Z","shell.execute_reply.started":"2022-06-24T04:17:32.062142Z","shell.execute_reply":"2022-06-24T04:17:32.077811Z"}}
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# %% [code] {"id":"sprLV8byHpfa","outputId":"eb57c916-7551-419d-e30d-fe315d8f0355","execution":{"iopub.status.busy":"2022-06-24T04:17:32.084628Z","iopub.execute_input":"2022-06-24T04:17:32.085136Z","iopub.status.idle":"2022-06-24T04:17:32.103604Z","shell.execute_reply.started":"2022-06-24T04:17:32.085098Z","shell.execute_reply":"2022-06-24T04:17:32.102215Z"}}
#we had 24 columns out of this we removed name and status column
# we want all  values in same range but it should not change the meaning the data conveys
# done using standardscaler
#standardise can be done before or after spliting dataset
scaler=StandardScaler()
scaler.fit(x_train)


# %% [code] {"id":"ZZf8P2a6JmcZ","outputId":"b5c74232-7e7e-4ca2-8eac-3fd7c2bbb22a","execution":{"iopub.status.busy":"2022-06-24T04:17:32.105698Z","iopub.execute_input":"2022-06-24T04:17:32.109317Z","iopub.status.idle":"2022-06-24T04:17:32.129848Z","shell.execute_reply.started":"2022-06-24T04:17:32.10925Z","shell.execute_reply":"2022-06-24T04:17:32.12723Z"}}
 #it transforms all  x_train values in same range
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
print(x_train)

# %% [code] {"id":"qgWJF5nZKF_b","execution":{"iopub.status.busy":"2022-06-24T04:17:32.132189Z","iopub.execute_input":"2022-06-24T04:17:32.136409Z","iopub.status.idle":"2022-06-24T04:17:32.149094Z","shell.execute_reply.started":"2022-06-24T04:17:32.13634Z","shell.execute_reply":"2022-06-24T04:17:32.147605Z"}}
#Implement SVM
# out of all features
#feature 1 -x axis value
#feature 2 -y axis value
#svm finds a line(hyperplane) which best seperates the person with parkinson and without parkinson disease
# when new instance is given to svm it will tell of put the data point in the region and tell whether person has diease or not
# 2 features then svm will be 2 Dimension model
# we have 24 features so it will be 24 dimension model



# %% [code] {"id":"I7tgPQlCL17A","execution":{"iopub.status.busy":"2022-06-24T04:17:32.150737Z","iopub.execute_input":"2022-06-24T04:17:32.15231Z","iopub.status.idle":"2022-06-24T04:17:32.16115Z","shell.execute_reply.started":"2022-06-24T04:17:32.152266Z","shell.execute_reply":"2022-06-24T04:17:32.15975Z"}}
#load svm model to variable model
#svc-classification
#svr-regression
model=svm.SVC(kernel='linear')


# %% [code] {"id":"GgaNEHo_Mmej","outputId":"4829aa07-516e-4d0a-92a4-c1482ba56786","execution":{"iopub.status.busy":"2022-06-24T04:17:32.162912Z","iopub.execute_input":"2022-06-24T04:17:32.165008Z","iopub.status.idle":"2022-06-24T04:17:32.179605Z","shell.execute_reply.started":"2022-06-24T04:17:32.164966Z","shell.execute_reply":"2022-06-24T04:17:32.178283Z"}}
#training svm model with training data
# data will be fitted in model
model.fit(x_train,y_train)
filename = './saved_model/finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))




# %% [markdown] {"id":"fCP3AO5LNOmj"}
# Evaluation of Model 

# %% [code] {"id":"BLyqJRX6NTE7","outputId":"458365b2-252e-4813-d2b6-002133251851","execution":{"iopub.status.busy":"2022-06-24T04:17:32.181387Z","iopub.execute_input":"2022-06-24T04:17:32.182538Z","iopub.status.idle":"2022-06-24T04:17:32.191808Z","shell.execute_reply.started":"2022-06-24T04:17:32.182486Z","shell.execute_reply":"2022-06-24T04:17:32.19008Z"}}
#Acuracy score on training data
x_train_prediction=model.predict(x_train)
training_data_score=accuracy_score(y_train,x_train_prediction)
print("Accuracy score:",training_data_score)

# %% [markdown] {"id":"0M4RNeKGQ7NL"}
# Building predictive model
#  

# %% [code] {"id":"d9E2nNYURBn1","outputId":"0cc7d28d-901e-4cc5-acac-e0c00938e790","execution":{"iopub.status.busy":"2022-06-24T04:17:32.194944Z","iopub.execute_input":"2022-06-24T04:17:32.19548Z","iopub.status.idle":"2022-06-24T04:17:32.212461Z","shell.execute_reply.started":"2022-06-24T04:17:32.195441Z","shell.execute_reply":"2022-06-24T04:17:32.210459Z"}}
#when new instance is given model should predict whether person has PD or not
# take 1 full column from dataset and put inside parantheses
input_data=(116.676,137.871,111.366,0.00997,0.00009,0.00502,0.00698,0.01505,0.05492,0.517,0.02924,0.04005,0.03772,0.08771,0.01353,20.644,0.434969,0.819235,-4.117501,0.334147,2.405554,0.368975)
#changing input data into numpy array
input_data_as_numpy_array=np.asarray(input_data)
#reshape the numpy array
#if we dont give reshape the model will be expecting 157 all data
#using reshape we are specifying that we are giving 1 datapoint and we are expection one target value
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
#standardise the data
standard_data=scaler.transform(input_data_reshaped)

#predict
prediction=model.predict(standard_data)
print(prediction)
if (prediction[0]==0):
  print("HEALTHY")
else:
  print("PARKINSON")

# %% [code] {"id":"Qcn1Dt4QT7V2"}

