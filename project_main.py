import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage.filters import gaussian_filter
from scipy.fftpack import fft
import scipy
import seaborn as sns
import models as m

import sklearn.linear_model as lm
import tensorflow as tf
import sklearn.svm as svm

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence

import sklearn.preprocessing as pproc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import normalize

def custom_random_oversampling(X, y, sampling_strategy=0.5, random_state=None):
    np.random.seed(random_state)
    unique_classes, class_counts = np.unique(y, return_counts=True)
    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]

    # Calculate the number of samples to select from the minority class
    n_samples_minority = int(len(majority_indices) * sampling_strategy)
    selected_minority_indices = np.random.choice(minority_indices, size=n_samples_minority, replace=True)

    # Combine minority and majority samples
    X_resampled = np.concatenate((X[selected_minority_indices], X[majority_indices]), axis=0)
    y_resampled = np.concatenate((y[selected_minority_indices], y[majority_indices]), axis=0)

    # Shuffle the combined samples
    shuffle_indices = np.arange(len(X_resampled))
    np.random.shuffle(shuffle_indices)
    X_resampled = X_resampled[shuffle_indices]
    y_resampled = y_resampled[shuffle_indices]

    return X_resampled, y_resampled




# import the dataset
data_train = pd.read_csv(r'C:\Users\Rishabh\Desktop\MLT LAB\exoTrain.csv')
data_test = pd.read_csv(r'C:\Users\Rishabh\Desktop\MLT LAB\exoTest.csv')



#permute the dataset
data_train = np.random.permutation(np.asarray(data_train))
data_test = np.random.permutation(np.asarray(data_test))



#get the Label column and delate the class column and rescale
y1 = data_train[:,0]
y2 = data_test[:,0]

y_train = (y1-min(y1))/(max(y1)-min(y1))
y_test = (y2-min(y2))/(max(y2)-min(y2))

data_train = np.delete(data_train,1,1)
data_test = np.delete(data_test,1,1)



#print the light curve
time = np.arange(len(data_train[0])) * (36/60)  # time in hours

plt.figure(figsize=(20,5))
plt.title('Flux of star 10 with confirmed planet')
plt.ylabel('Flux')
plt.xlabel('Hours')
plt.plot( time , data_train[10] )     #change the number to plot what you want



#normalized data
data_train_norm = normalize(data_train)
data_test_norm = normalize(data_test)




# function to apply gaussian filter to all data
def gauss_filter(dataset,sigma):
    
    dts = []
    
    for x in range(dataset.shape[0]):
        dts.append(gaussian_filter(dataset[x], sigma))
    
    return np.asarray(dts)



# apply the gaussian filter to all rows data
data_train_gaussian = gauss_filter(data_train_norm,7.0)
data_test_gaussian = gauss_filter(data_test_norm,7.0)




#print the light curves smoothed
plt.figure(figsize=(20,5))
plt.title('Flux of star 10 with confirmed planet, smoothed')
plt.ylabel('Flux')
plt.xlabel('Hours')
#plt.plot( time , data_train_gaussian[1000])




# apply FFT to the data smoothed
frequency = np.arange(len(data_train[0])) * (1/(36.0*60.0))

data_train_fft1 = scipy.fft.fft2(data_train_norm, axes=1)
data_test_fft1 = scipy.fft.fft2(data_test_norm, axes=1)

data_train_fft = np.abs(data_train_fft1)   #calculate the abs value
data_test_fft = np.abs(data_test_fft1)



#get the length of the FFT data, make something here below in order to make the sequences of the same size
# only if they have differet dimensions

len_seq = len(data_train_fft[0])



#plot the FFT of the signals
plt.figure(figsize=(20,5))
plt.title('flux of star 1 ( with confirmed planet ) in domain of frequencies')
plt.ylabel('Abs value of FFT result')
plt.xlabel('Frequency')
plt.plot( frequency, data_train_fft[1] )



#oversampling technique to the data
data_train_ovs, y_train_ovs = custom_random_oversampling(data_train_fft, y_train, sampling_strategy=0.5, random_state=42)




#recap dataset after oversampling
print("After oversampling, counts of label '1': {}".format(sum(y_train_ovs==1)))
print("After oversampling, counts of label '0': {}".format(sum(y_train_ovs==0)))



#reshape the data for the neural network model
data_train_ovs = np.asarray(data_train_ovs)
data_test_fft = np.asarray(data_test_fft)

data_train_ovs_nn = data_train_ovs.reshape((data_train_ovs.shape[0], data_train_ovs.shape[1], 1))
data_test_fft_nn = data_test_fft.reshape((data_test_fft.shape[0], data_test_fft.shape[1], 1))



#create F.C.N model and run it
model = m.FCN_model(len_seq)

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])

print(model.summary())

history = model.fit(data_train_ovs_nn, y_train_ovs , epochs=15, batch_size = 10, validation_data=(data_test_fft_nn, y_test))


#save the model if you want
#model.save("/home/senad/DataSet/exoplanet_model_2")


#load the model if already exsist
#model = tf.keras.models.load_model("/home/senad/DataSet/exoplanet_model_2")


acc = history.history['accuracy']
#acc_val = history.history['val_accuracy']
epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'b', label='accuracy_train')
#plt.plot(epochs, acc_val, 'g', label='accuracy_val')
plt.title('accuracy')
plt.xlabel('epochs')
plt.ylabel('value of accuracy')
plt.legend()
plt.grid()
plt.show()




loss = history.history['loss']
#loss_val = history.history['val_loss']
epochs = range(1, len(acc)+1)
plt.plot(epochs, loss, 'b', label='loss_train')
#plt.plot(epochs, loss_val, 'g', label='loss_val')
plt.title('loss')
plt.xlabel('epochs')
plt.ylabel('value of loss')
plt.legend()
plt.grid()
plt.show()




#predict the test set and plot results
y_test_pred = model.predict(data_test_fft_nn)
y_test_pred = (y_test_pred > 0.5)


accuracy = accuracy_score(y_test, y_test_pred)
print("accuracy : ", accuracy)

print(classification_report(y_test, y_test_pred, target_names=["NO exoplanet confirmed","YES exoplanet confirmed"]))

conf_matrix = confusion_matrix([int(x) for x in y_test ], [int(y) for y in y_test_pred ])
sns.heatmap(conf_matrix, annot=True, cmap='Blues')




#create SVC model, predict and plot the results
SVC = m.SVC_model()
SVC.fit(data_train_ovs, y_train_ovs)

y_pred_svc = SVC.predict(data_test_fft)


print(classification_report(y_test, y_pred_svc, target_names=["NO exoplanet confirmed","YES exoplanet confirmed"]))


conf_matrix = confusion_matrix([int(x) for x in y_test ], [int(y) for y in y_pred_svc ])
sns.heatmap(conf_matrix, annot=True, cmap='Blues')


