# %% [code] {"papermill":{"duration":5.579884,"end_time":"2021-04-25T07:07:55.709182","exception":false,"start_time":"2021-04-25T07:07:50.129298","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:44:02.767521Z","iopub.execute_input":"2022-06-15T05:44:02.768054Z","iopub.status.idle":"2022-06-15T05:44:09.740519Z","shell.execute_reply.started":"2022-06-15T05:44:02.767968Z","shell.execute_reply":"2022-06-15T05:44:09.739721Z"}}
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split

# %% [code] {"execution":{"iopub.status.busy":"2022-06-15T05:44:09.741631Z","iopub.execute_input":"2022-06-15T05:44:09.742048Z","iopub.status.idle":"2022-06-15T05:44:11.909536Z","shell.execute_reply.started":"2022-06-15T05:44:09.742017Z","shell.execute_reply":"2022-06-15T05:44:11.908399Z"}}
#!pip freeze

# %% [code] {"papermill":{"duration":0.022612,"end_time":"2021-04-25T07:07:55.749535","exception":false,"start_time":"2021-04-25T07:07:55.726923","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:44:11.911919Z","iopub.execute_input":"2022-06-15T05:44:11.912311Z","iopub.status.idle":"2022-06-15T05:44:11.917204Z","shell.execute_reply.started":"2022-06-15T05:44:11.912265Z","shell.execute_reply":"2022-06-15T05:44:11.916171Z"}}
dir_sp_train = './archive/drawings/spiral/training'
dir_sp_test = './archive/drawings/spiral/testing'
dir_wv_train = './archive/drawings/wave/training'
dir_wv_test = './archive/drawings/wave/testing'

# %% [code] {"papermill":{"duration":0.15222,"end_time":"2021-04-25T07:07:55.917393","exception":false,"start_time":"2021-04-25T07:07:55.765173","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:44:11.918620Z","iopub.execute_input":"2022-06-15T05:44:11.918959Z","iopub.status.idle":"2022-06-15T05:44:11.940375Z","shell.execute_reply.started":"2022-06-15T05:44:11.918930Z","shell.execute_reply":"2022-06-15T05:44:11.939375Z"}}
Name=[]     
for file in os.listdir(dir_sp_train):
    Name+=[file]
print(Name)
print(len(Name))

# %% [code] {"papermill":{"duration":0.023531,"end_time":"2021-04-25T07:07:55.957708","exception":false,"start_time":"2021-04-25T07:07:55.934177","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:44:11.941584Z","iopub.execute_input":"2022-06-15T05:44:11.941849Z","iopub.status.idle":"2022-06-15T05:44:11.946316Z","shell.execute_reply.started":"2022-06-15T05:44:11.941822Z","shell.execute_reply":"2022-06-15T05:44:11.945614Z"}}
N=[]
for i in range(len(Name)):
    N+=[i]
    
normal_mapping=dict(zip(Name,N)) 
reverse_mapping=dict(zip(N,Name)) 

def mapper(value):
    return reverse_mapping[value]

# %% [code] {"papermill":{"duration":14.785601,"end_time":"2021-04-25T07:08:10.798527","exception":false,"start_time":"2021-04-25T07:07:56.012926","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:44:11.947465Z","iopub.execute_input":"2022-06-15T05:44:11.947725Z","iopub.status.idle":"2022-06-15T05:44:12.971793Z","shell.execute_reply.started":"2022-06-15T05:44:11.947689Z","shell.execute_reply":"2022-06-15T05:44:12.970839Z"}}
dataset_sp=[]
count=0
for file in os.listdir(dir_sp_train):
    path=os.path.join(dir_sp_train,file)
    for im in os.listdir(path):
        image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(100,100))
        image=img_to_array(image)
        image=image/255.0
        dataset_sp.append([image,count])
    count=count+1
    
testset_sp=[]
count=0
for file in os.listdir(dir_sp_test):
    path=os.path.join(dir_sp_test,file)
    for im in os.listdir(path):
        image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(100,100))
        image=img_to_array(image)
        image=image/255.0
        testset_sp.append([image,count])
    count=count+1    

# %% [code] {"execution":{"iopub.status.busy":"2022-06-15T05:44:12.973038Z","iopub.execute_input":"2022-06-15T05:44:12.973301Z","iopub.status.idle":"2022-06-15T05:44:14.161075Z","shell.execute_reply.started":"2022-06-15T05:44:12.973267Z","shell.execute_reply":"2022-06-15T05:44:14.160074Z"}}
dataset_wv=[]
count=0
for file in os.listdir(dir_wv_train):
    path=os.path.join(dir_wv_train,file)
    for im in os.listdir(path):
        image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(100,100))
        image=img_to_array(image)
        image=image/255.0
        dataset_wv.append([image,count])
    count=count+1
    
testset_wv=[]
count=0
for file in os.listdir(dir_wv_test):
    path=os.path.join(dir_wv_test,file)
    for im in os.listdir(path):
        image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(100,100))
        image=img_to_array(image)
        image=image/255.0
        testset_wv.append([image,count])
    count=count+1   

# %% [code] {"papermill":{"duration":0.164791,"end_time":"2021-04-25T07:08:10.980431","exception":false,"start_time":"2021-04-25T07:08:10.81564","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:44:14.163605Z","iopub.execute_input":"2022-06-15T05:44:14.163931Z","iopub.status.idle":"2022-06-15T05:44:14.168295Z","shell.execute_reply.started":"2022-06-15T05:44:14.163898Z","shell.execute_reply":"2022-06-15T05:44:14.167285Z"}}
data_sp,labels_sp0=zip(*dataset_sp)
test_sp,tlabels_sp0=zip(*testset_sp)

data_wv,labels_wv0=zip(*dataset_wv)
test_wv,tlabels_wv0=zip(*testset_wv)

# %% [code] {"papermill":{"duration":0.137825,"end_time":"2021-04-25T07:08:11.135337","exception":false,"start_time":"2021-04-25T07:08:10.997512","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:44:14.170144Z","iopub.execute_input":"2022-06-15T05:44:14.170534Z","iopub.status.idle":"2022-06-15T05:44:14.192871Z","shell.execute_reply.started":"2022-06-15T05:44:14.170501Z","shell.execute_reply":"2022-06-15T05:44:14.191505Z"}}
labels_sp1=to_categorical(labels_sp0)
data_sp=np.array(data_sp)
labels_sp=np.array(labels_sp1)

tlabels_sp1=to_categorical(tlabels_sp0)
test_sp=np.array(test_sp)
tlabels_sp=np.array(tlabels_sp1)

# %% [code] {"papermill":{"duration":0.123877,"end_time":"2021-04-25T07:08:11.277006","exception":false,"start_time":"2021-04-25T07:08:11.153129","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:44:14.194321Z","iopub.execute_input":"2022-06-15T05:44:14.194611Z","iopub.status.idle":"2022-06-15T05:44:14.208598Z","shell.execute_reply.started":"2022-06-15T05:44:14.194582Z","shell.execute_reply":"2022-06-15T05:44:14.207236Z"}}
labels_wv1=to_categorical(labels_wv0)
data_wv=np.array(data_wv)
labels_wv=np.array(labels_wv1)

tlabels_wv1=to_categorical(tlabels_wv0)
test_wv=np.array(test_wv)
tlabels_wv=np.array(tlabels_wv1)

# %% [code] {"papermill":{"duration":0.120415,"end_time":"2021-04-25T07:08:11.460311","exception":false,"start_time":"2021-04-25T07:08:11.339896","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:44:14.210011Z","iopub.execute_input":"2022-06-15T05:44:14.210323Z","iopub.status.idle":"2022-06-15T05:44:14.228507Z","shell.execute_reply.started":"2022-06-15T05:44:14.210292Z","shell.execute_reply":"2022-06-15T05:44:14.227661Z"}}
trainx_sp,testx_sp,trainy_sp,testy_sp=train_test_split(data_sp,labels_sp,test_size=0.2,random_state=44)
trainx_wv,testx_wv,trainy_wv,testy_wv=train_test_split(data_wv,labels_wv,test_size=0.2,random_state=44)

# %% [code] {"papermill":{"duration":0.027242,"end_time":"2021-04-25T07:08:11.506602","exception":false,"start_time":"2021-04-25T07:08:11.47936","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:44:14.229689Z","iopub.execute_input":"2022-06-15T05:44:14.230227Z","iopub.status.idle":"2022-06-15T05:44:14.235848Z","shell.execute_reply.started":"2022-06-15T05:44:14.230168Z","shell.execute_reply":"2022-06-15T05:44:14.234989Z"}}
print(trainx_sp.shape)
print(testx_sp.shape)
print(trainy_sp.shape)
print(testy_sp.shape)

# %% [code] {"papermill":{"duration":0.025748,"end_time":"2021-04-25T07:08:11.55027","exception":false,"start_time":"2021-04-25T07:08:11.524522","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:44:14.237195Z","iopub.execute_input":"2022-06-15T05:44:14.237567Z","iopub.status.idle":"2022-06-15T05:44:14.246690Z","shell.execute_reply.started":"2022-06-15T05:44:14.237518Z","shell.execute_reply":"2022-06-15T05:44:14.245879Z"}}
datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=20,zoom_range=0.2,
                        width_shift_range=0.2,height_shift_range=0.2,shear_range=0.1,fill_mode="nearest")

# %% [code] {"papermill":{"duration":5.516588,"end_time":"2021-04-25T07:08:17.08603","exception":false,"start_time":"2021-04-25T07:08:11.569442","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:44:14.249231Z","iopub.execute_input":"2022-06-15T05:44:14.249664Z","iopub.status.idle":"2022-06-15T05:44:24.933131Z","shell.execute_reply.started":"2022-06-15T05:44:14.249618Z","shell.execute_reply":"2022-06-15T05:44:24.932375Z"}}
pretrained_model3 = tf.keras.applications.DenseNet201(input_shape=(100,100,3),include_top=False,weights='imagenet',pooling='avg')
pretrained_model3.trainable = False

pretrained_model4 = tf.keras.applications.DenseNet201(input_shape=(100,100,3),include_top=False,weights='imagenet',pooling='avg')
pretrained_model4.trainable = False

# %% [code] {"papermill":{"duration":0.063718,"end_time":"2021-04-25T07:08:17.169902","exception":false,"start_time":"2021-04-25T07:08:17.106184","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:44:24.934609Z","iopub.execute_input":"2022-06-15T05:44:24.935310Z","iopub.status.idle":"2022-06-15T05:44:25.059401Z","shell.execute_reply.started":"2022-06-15T05:44:24.935254Z","shell.execute_reply":"2022-06-15T05:44:25.058226Z"}}
inputs3 = pretrained_model3.input
x3 = tf.keras.layers.Dense(128, activation='relu')(pretrained_model3.output)
outputs3 = tf.keras.layers.Dense(2, activation='softmax')(x3)
model3 = tf.keras.Model(inputs=inputs3, outputs=outputs3)

inputs4 = pretrained_model4.input
x4 = tf.keras.layers.Dense(128, activation='relu')(pretrained_model4.output)
outputs4 = tf.keras.layers.Dense(2, activation='softmax')(x4)
model4 = tf.keras.Model(inputs=inputs4, outputs=outputs4)

# %% [code] {"papermill":{"duration":0.041368,"end_time":"2021-04-25T07:08:17.231582","exception":false,"start_time":"2021-04-25T07:08:17.190214","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:44:25.060689Z","iopub.execute_input":"2022-06-15T05:44:25.061026Z","iopub.status.idle":"2022-06-15T05:44:25.161168Z","shell.execute_reply.started":"2022-06-15T05:44:25.060993Z","shell.execute_reply":"2022-06-15T05:44:25.160093Z"}}
model3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model4.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# %% [code] {"papermill":{"duration":369.170719,"end_time":"2021-04-25T07:14:26.422069","exception":false,"start_time":"2021-04-25T07:08:17.25135","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:44:25.162637Z","iopub.execute_input":"2022-06-15T05:44:25.163070Z","iopub.status.idle":"2022-06-15T05:47:44.751225Z","shell.execute_reply.started":"2022-06-15T05:44:25.163026Z","shell.execute_reply":"2022-06-15T05:47:44.750417Z"}}

# ------------ train wave modal ------------
checkpoint_wave_path='./wave/cp-{epoch:04d}.ckpt'
checkpoint_wave_dir = os.path.dirname(checkpoint_wave_path)
cp_wave_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_wave_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model4.save_weights(checkpoint_wave_path.format(epoch=0))                                                 
latest = tf.train.latest_checkpoint(checkpoint_wave_dir)
model4.load_weights(latest)

his4=model4.fit(datagen.flow(trainx_wv,trainy_wv,batch_size=32),validation_data=(testx_wv,testy_wv),epochs=10, callbacks=[cp_wave_callback])

model4.save('./saved_model/model_wave')
# ------------ train wave modal ends here ------------




# ------------ train spiral modal ------------

checkpoint_spiral_path='./spiral/cp-{epoch:04d}.ckpt'
checkpoint_spiral_dir = os.path.dirname(checkpoint_spiral_path)
cp_spiral_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_spiral_path,
                                                 save_weights_only=True,
                                                 verbose=1)
model3.save_weights(checkpoint_spiral_path.format(epoch=0))
latest = tf.train.latest_checkpoint(checkpoint_spiral_dir)
model3.load_weights(latest)
his3=model3.fit(datagen.flow(trainx_sp,trainy_sp,batch_size=32),validation_data=(testx_sp,testy_sp),epochs=10, callbacks=[cp_spiral_callback])

model3.save('./saved_model/model_spiral')

# ------------ train wave modal ends here ------------



# %% [code] {"papermill":{"duration":3.53915,"end_time":"2021-04-25T07:14:30.789745","exception":false,"start_time":"2021-04-25T07:14:27.250595","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:47:44.755040Z","iopub.execute_input":"2022-06-15T05:47:44.755992Z","iopub.status.idle":"2022-06-15T05:47:48.230777Z","shell.execute_reply.started":"2022-06-15T05:47:44.755940Z","shell.execute_reply":"2022-06-15T05:47:48.229765Z"}}
#spiral
y_pred_sp=model3.predict(testx_sp)
pred_sp=np.argmax(y_pred_sp,axis=1)
ground_sp = np.argmax(testy_sp,axis=1)
print(classification_report(ground_sp,pred_sp))

# %% [code] {"execution":{"iopub.status.busy":"2022-06-15T05:47:48.232117Z","iopub.execute_input":"2022-06-15T05:47:48.232665Z","iopub.status.idle":"2022-06-15T05:47:48.644598Z","shell.execute_reply.started":"2022-06-15T05:47:48.232621Z","shell.execute_reply":"2022-06-15T05:47:48.642873Z"}}
#wave
y_pred_wv=model3.predict(testx_wv)
pred_wv=np.argmax(y_pred_wv,axis=1)
ground_wv = np.argmax(testy_wv,axis=1)
print(classification_report(ground_wv,pred_wv))

# %% [code] {"papermill":{"duration":0.995914,"end_time":"2021-04-25T07:14:32.648575","exception":false,"start_time":"2021-04-25T07:14:31.652661","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:47:48.646353Z","iopub.execute_input":"2022-06-15T05:47:48.646776Z","iopub.status.idle":"2022-06-15T05:47:48.981885Z","shell.execute_reply.started":"2022-06-15T05:47:48.646715Z","shell.execute_reply":"2022-06-15T05:47:48.981022Z"}}
get_acc3 = his3.history['accuracy']
value_acc3 = his3.history['val_accuracy']
get_loss3 = his3.history['loss']
validation_loss3 = his3.history['val_loss']

epochs3 = range(len(get_acc3))
plt.plot(epochs3, get_acc3, 'r', label='Accuracy of Training data')
plt.plot(epochs3, value_acc3, 'b', label='Accuracy of Validation data')
plt.title('Training vs validation accuracy - Spiral')
plt.legend(loc=0)
plt.figure()
plt.show()

epochs3 = range(len(get_loss3))
plt.plot(epochs3, get_loss3, 'r', label='Loss of Training data')
plt.plot(epochs3, validation_loss3, 'b', label='Loss of Validation data')
plt.title('Training vs validation loss - Spiral')
plt.legend(loc=0)
plt.figure()
plt.show()

# %% [code] {"papermill":{"duration":0.971858,"end_time":"2021-04-25T07:14:34.468705","exception":false,"start_time":"2021-04-25T07:14:33.496847","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:47:48.983173Z","iopub.execute_input":"2022-06-15T05:47:48.983469Z","iopub.status.idle":"2022-06-15T05:47:49.289919Z","shell.execute_reply.started":"2022-06-15T05:47:48.983439Z","shell.execute_reply":"2022-06-15T05:47:49.288902Z"}}
get_acc4 = his4.history['accuracy']
value_acc4 = his4.history['val_accuracy']
get_loss4 = his4.history['loss']
validation_loss4 = his4.history['val_loss']

epochs4 = range(len(get_acc4))
plt.plot(epochs4, get_acc4, 'r', label='Accuracy of Training data')
plt.plot(epochs4, value_acc4, 'b', label='Accuracy of Validation data')
plt.title('Training vs validation accuracy - Wave')
plt.legend(loc=0)
plt.figure()
plt.show()

epochs4 = range(len(get_loss4))
plt.plot(epochs4, get_loss4, 'r', label='Loss of Training data')
plt.plot(epochs4, validation_loss4, 'b', label='Loss of Validation data')
plt.title('Training vs validation loss - Wave')
plt.legend(loc=0)
plt.figure()
plt.show()

# %% [code] {"papermill":{"duration":0.857299,"end_time":"2021-04-25T07:14:36.189773","exception":false,"start_time":"2021-04-25T07:14:35.332474","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:47:49.291242Z","iopub.execute_input":"2022-06-15T05:47:49.291512Z","iopub.status.idle":"2022-06-15T05:47:49.312529Z","shell.execute_reply.started":"2022-06-15T05:47:49.291485Z","shell.execute_reply":"2022-06-15T05:47:49.311722Z"}}
load_img("../input/parkinsons-drawings/spiral/testing/parkinson/V03PE07.png",target_size=(100,100))


# %% [code] {"papermill":{"duration":0.887719,"end_time":"2021-04-25T07:14:37.940876","exception":false,"start_time":"2021-04-25T07:14:37.053157","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:47:49.314797Z","iopub.execute_input":"2022-06-15T05:47:49.315064Z","iopub.status.idle":"2022-06-15T05:47:49.424531Z","shell.execute_reply.started":"2022-06-15T05:47:49.315038Z","shell.execute_reply":"2022-06-15T05:47:49.423529Z"}}
image=load_img("../input/parkinsons-drawings/spiral/testing/parkinson/V03PE07.png",target_size=(100,100))

image=img_to_array(image) 
image=image/255.0
prediction_image=np.array(image)
prediction_image=np.expand_dims(image, axis=0)

prediction=model3.predict(prediction_image)
value=np.argmax(prediction)
move_name=mapper(value)
print("Prediction is {}.".format(move_name))

# %% [code] {"execution":{"iopub.status.busy":"2022-06-15T05:47:49.425631Z","iopub.execute_input":"2022-06-15T05:47:49.425940Z","iopub.status.idle":"2022-06-15T05:47:49.447241Z","shell.execute_reply.started":"2022-06-15T05:47:49.425909Z","shell.execute_reply":"2022-06-15T05:47:49.446314Z"}}
load_img("../input/parkinsons-drawings/wave/testing/parkinson/V03PO01.png",target_size=(100,100))

# %% [code] {"papermill":{"duration":1.22121,"end_time":"2021-04-25T07:14:41.900362","exception":false,"start_time":"2021-04-25T07:14:40.679152","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:47:49.448565Z","iopub.execute_input":"2022-06-15T05:47:49.448848Z","iopub.status.idle":"2022-06-15T05:47:53.078560Z","shell.execute_reply.started":"2022-06-15T05:47:49.448820Z","shell.execute_reply":"2022-06-15T05:47:53.077232Z"}}
image2=load_img("../input/parkinsons-drawings/wave/testing/parkinson/V03PO01.png",target_size=(100,100))

image2=img_to_array(image2) 
image2=image2/255.0
prediction_image2=np.array(image2)
prediction_image2=np.expand_dims(image2, axis=0)

prediction2=model4.predict(prediction_image2)
value2=np.argmax(prediction2)
move_name2=mapper(value2)
print("Prediction is {}.".format(move_name2))


# %% [code] {"papermill":{"duration":2.585739,"end_time":"2021-04-25T07:14:45.326559","exception":false,"start_time":"2021-04-25T07:14:42.74082","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:47:53.079806Z","iopub.execute_input":"2022-06-15T05:47:53.080424Z","iopub.status.idle":"2022-06-15T05:47:53.768851Z","shell.execute_reply.started":"2022-06-15T05:47:53.080379Z","shell.execute_reply":"2022-06-15T05:47:53.768035Z"}}
print(test_sp.shape)
prediction_sp=model3.predict(test_sp)
print(prediction_sp.shape)

PRED_sp=[]
for item in prediction_sp:
    value_sp=np.argmax(item)      
    PRED_sp+=[value_sp]
    
ANS_sp=tlabels_sp0
accuracy_sp=accuracy_score(ANS_sp,PRED_sp)
print(accuracy_sp)    

# %% [code] {"papermill":{"duration":0.841886,"end_time":"2021-04-25T07:14:47.000496","exception":false,"start_time":"2021-04-25T07:14:46.15861","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-15T05:47:53.771575Z","iopub.execute_input":"2022-06-15T05:47:53.772915Z","iopub.status.idle":"2022-06-15T05:47:54.431492Z","shell.execute_reply.started":"2022-06-15T05:47:53.772877Z","shell.execute_reply":"2022-06-15T05:47:54.430721Z"}}
print(test_wv.shape)
prediction_wv=model4.predict(test_wv)
print(prediction_wv.shape)

PRED_wv=[]
for item in prediction_wv:
    value_wv=np.argmax(item)      
    PRED_wv+=[value_wv]
    
ANS_wv=tlabels_wv0
accuracy_wv=accuracy_score(ANS_wv,PRED_wv)
print(accuracy_wv)    

# %% [code]


# %% [code]

