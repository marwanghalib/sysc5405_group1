#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:29:58 2021

@author: PalwashaW_Shaikh
"""

#SIMPLE RNN Binary Classification with Regression

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import set_printoptions
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score,roc_auc_score
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import spearmanr, pearsonr
import itertools as IT
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from scipy import interp
from featurewiz import featurewiz
from sklearn.feature_selection import RFE,RFECV
from sklearn.tree import DecisionTreeClassifier # Import 


#READ TRAINING DATA SET PROVIDED

df = pd.read_csv ('train_data.csv')#header=None)
print(df)
print(df.head())

X_feat=df.iloc[:,0:336] #separate into feature and target data - removed KIBA and target
Y_target=df.loc[:,"KIBA"] #KIBA
Y_target=pd.DataFrame(Y_target)
print(X_feat)
print(Y_target)

#PRE_PROCESSING WITH MINMAX Scaler
X=pd.DataFrame(X_feat)
scaler = MinMaxScaler()
X_feat_t=scaler.fit_transform(X)
X_feat_s=pd.DataFrame(X_feat_t) #Scaled features
print(X_feat_s)

#Scaled KIBA score
y=scaler.fit_transform(Y_target)




#ANOVA F
# feature extraction
test2 = SelectKBest(score_func=f_classif, k=5) # 5 best features 
fit2 = test2.fit(X_feat_s, y)
# summarize scores
set_printoptions(precision=3)
#print(fit2.scores_)

#Transform features
X_feat_new2 = fit2.transform(X_feat)

df3=df.iloc[:,0:336]
# LIST OF COLUMNS SELECTED --- use for test data as well later on
anova_col_names=df3.columns[test2.get_support()] 
print("List of ANOVA-F selected feature names")
print(anova_col_names)

# summarize selected features
print(X_feat_new2[0:5,:])
print(X_feat_new2)
dfscores2 = pd.DataFrame(fit2.scores_)
dfcolumns2 = pd.DataFrame(X_feat.columns)
#concat two dataframes for better visualization 
featureScores2 = pd.concat([dfcolumns2,dfscores2],axis=1)
featureScores2.columns = ['Feature','Score ANOVA F']  #naming the dataframe columns
print(featureScores2.nlargest(25,'Score ANOVA F'))
 
#Plot top 25 features for visualization
f2=featureScores2.nlargest(25,'Score ANOVA F')
f2.plot.bar(x="Feature", y="Score ANOVA F", rot=0)
plt.xticks(rotation=90)
plt.show()
plt.clf()



#Pre-processed or scaled and selected 5 features
X_s_t=pd.DataFrame(X_feat_new2)

#Pre-pocessed + feature slected data 
data_x=X_s_t
data_y=y

#split into train test - hold out
X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.25,random_state=123) #stratified

print("Shape of Train-Test Split")
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

#Build a simple RNN ___________________________________

#transform data into numpy array 
x_training_data=np.array(X_train)
y_training_data=np.array(Y_train)
x_test_data=np.array(X_test)
y_test_data=np.array(Y_test)

#check data
# print(x_training_data)
# print(y_training_data)
# print(x_test_data)
# print(y_test_data)

#Verifying the shape of the NumPy arrays
print(x_training_data.shape)
print(y_training_data.shape)
print(x_test_data.shape)
print(y_test_data.shape)


#Reshaping the NumPy array to meet TensorFlow standards
x_training_data = np.reshape(x_training_data, (x_training_data.shape[0],  x_training_data.shape[1], 1))
x_test_data = np.reshape(x_test_data, (x_test_data.shape[0],  x_test_data.shape[1], 1))

#Printing the new shape 
print(x_training_data.shape)
print(x_test_data.shape)
#Importing our TensorFlow libraries

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LSTM, SimpleRNN

from tensorflow.keras.layers import Dropout

#Initializing our recurrent neural network

simp_rnn = Sequential()

#Layer 1 #(x_training_data.shape[1], 1)
simp_rnn.add(SimpleRNN(units=256, return_sequences = True, input_shape = (x_training_data.shape[1], 1)))
simp_rnn.add(Dropout(0.2))
simp_rnn.add(SimpleRNN(units=256, return_sequences = True))
simp_rnn.add(Dropout(0.2))
simp_rnn.add(SimpleRNN(units=256, return_sequences = False))
simp_rnn.add(Dropout(0.2))
#Output Layer 
simp_rnn.add(Dense(units = 1,activation ='linear'))#LINEAR

simp_rnn.summary()

#Compiling the recurrent neural network

simp_rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')

from keras.callbacks import EarlyStopping, ModelCheckpoint


# Create callbacks
callbacks=[]
es=EarlyStopping(monitor='val_loss', patience=5)
callbacks.append(es)
mcp=ModelCheckpoint('../simp_rnn_models/simp_rnn_model.h5', save_best_only=True, save_weights_only=False)
callbacks.append(mcp)

#Training the recurrent neural network

simp_history=simp_rnn.fit(x_training_data, y_training_data, epochs = 100, batch_size = 32, callbacks=callbacks, validation_data=(x_test_data, y_test_data),verbose=True)


test_loss = simp_rnn.evaluate(x_test_data, y_test_data)
print('Test loss:', test_loss)


# evaluate the model
train_loss = simp_rnn.evaluate(x_training_data,y_training_data, verbose=0)
test_loss = simp_rnn.evaluate(x_test_data, y_test_data, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_loss, test_loss))



#plot training history
plt.clf()
plt.title("Simple RNN Loss")
plt.plot(simp_history.history['loss'], label='train')
plt.plot(simp_history.history['val_loss'], label='test')
plt.legend()
plt.show()



#Generating our prediction probabilites for KIBA SCORES 

simp_rnn_preda = simp_rnn.predict(x_test_data).ravel()
print(simp_rnn_preda)


#Get predicition values ---- EXACT KIBA SCORE VALUES PREDICTED
simp_rnn_rawpreda = simp_rnn.predict(x_test_data)
simp_rnn_predClass=simp_rnn_rawpreda#tf.greater(simp_rnn_rawpreda, .5)
print(simp_rnn_predClass)
unscaled_predictions = scaler.inverse_transform(simp_rnn_predClass)
predictions = unscaled_predictions
print(predictions)



#Plotting our predicted values

plt.clf() #This clears the old plot from our canvas


#Plotting the predicted values against actual 

plt.plot(predictions, color = '#135485', label = "Predictions")
plt.show()

plt.title('DTI Predictions')
plt.clf()

#unscale y test data labels
unscaled_ytest = scaler.inverse_transform(y_test_data)
print(unscaled_ytest)
plt.plot(unscaled_ytest, color = 'black', label = "Real Data")
plt.title('DTI Actual')
plt.show()


print(predictions)
print(unscaled_ytest)
print("Difference between True and Predicted KIBA")
print(np.subtract(unscaled_ytest, predictions))

# simp_rnn_predClass=[]
# #get labels to compare for fun KIBA >12.1 True 
# for x in predictions:
#     if(x>12.1):
#         simp_rnn_predClass.append("True")
#     else:
#         simp_rnn_predClass.append("False")
  

