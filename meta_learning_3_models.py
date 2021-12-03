#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 08:23:35 2021
@author: PalwashaW_Shaikh
"""


# RNN WITH GRU - 5 ANOVA F features - Binary Class.____________________________________________________________________
import pandas as pd
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import set_printoptions
from sklearn import tree
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
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from scipy.stats import spearmanr, pearsonr
import itertools as IT
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from scipy import interp
from sklearn.feature_selection import RFE,RFECV
#Importing our TensorFlow libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, SimpleRNN
from tensorflow.keras.layers import Dropout, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

from sklearn.datasets import make_regression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.datasets import make_classification
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
# define dataset


model_GRU = tf.keras.models.load_model('gru_rnn_model_last.h5')
model_LSTM = tf.keras.models.load_model('lstm_rnn_model_last.h5')

def print_confusion_matrix(Y_test,Y_pred):
    #compute confusion matrix, print it and create heat map then return confusion matrix
    cm = confusion_matrix(Y_test, Y_pred)
    print('Confusion matrix\n\n', cm)
    print('True Positives(TP) = ', cm[1,1])
    print('True Negatives(TN) = ', cm[0,0])
    print('False Positives(FP) = ', cm[0,1])
    print('False Negatives(FN) = ', cm[1,0])
    # visualize confusion matrix with seaborn heatmap
    plt.clf()
    plt.figure()
    cm_matrix = pd.DataFrame(data=cm, index=['Actual Neg:0','Actual Pos:1'], columns=['Predict Neg:0', 'Predict Pos:1'])
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    plt.savefig('gru_rnn_confusion_matrix.png')
    plt.show()
    return cm

def compute_cm_stat(cm):
    #Takes a confusion matrx and prints all results
    TP = cm[1,1]
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    # print classification accuracy
    classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
    #print('Classification Accuracy : {0:0.4f}'.format(classification_accuracy))
    print(f"Classification Accuracy : {classification_accuracy:.5f}")
    #print('Classification error : {0:0.4f}'.format(classification_error))
    # print classification error
    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    print(f'Classification Error : {classification_error:.5f}')
    # print precision score
    precision = TP / float(TP + FP)
    #print('Precision : {0:0.4f}'.format(precision))
    print(f'Precision : {precision:.5f}')
    # print recall
    recall = TP / float(TP + FN)
    #print('Recall or Sensitivity or True Pos. Rate : {0:0.4f}'.format(recall))
    print(f'Recall or Sensitivity or True Pos. Rate : {recall:.5f}')
    # print Sp
    specificity = TN / (TN + FP)
    #print('Specificity : {0:0.4f}'.format(specificity))
    print(f'Specificity : {specificity:.5f}')
    # print FPR
    false_positive_rate = FP / float(FP + TN)
    #print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
    print(f'False Positive Rate : {false_positive_rate:.5f}')
    # print FNR
    false_neg_rate = FN / float(TP + FN)
    #print('False Negative Rate : {0:0.4f}'.format(false_neg_rate))
    print(f'False Negative Rate : {false_neg_rate:.5f}')
    # print NPV
    neg_pred_val = TN / float(TN + FN)
    #print('Negative Predictive Value : {0:0.4f}'.format(neg_pred_val))
    print(f'Negative Predictive Value : {neg_pred_val:.5f}')
    # print FDR
    fdr = FP / float(TP + FP)
    #print('False Discovery Rate : {0:0.4f}'.format(fdr))
    print(f'False Discovery Rate : {fdr:.5f}')
    
    return{"acc":classification_accuracy,"ce":classification_error,
           "pr":precision, "re":recall, "sp":specificity, "fpr":false_positive_rate,
           "fnr":false_neg_rate,"npv":neg_pred_val,"fdr":fdr
           }

def plot_roc_CV(fpr_cv,tpr_cv,aucs,ns_fpr,ns_tpr,ns_auc,mean_tpr,mean_fpr,mean_auc,g_title):
    plt.figure(figsize=(6,4))
    for k in range(len(fpr_cv)):
        #plot fold ROCs
        plt.plot(fpr_cv[k], tpr_cv[k], lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.5f)' % (k+1, aucs[k]))
        
   
    #plot avg 5-fold CV classifier
    plt.plot(mean_fpr, mean_tpr, color='blue', label=r'Mean ROC (AUC = %0.5f )' % (mean_auc),lw=2, alpha=1)
    #plt.rcParams['font.size'] = 12
    
    #plot rnd classifier
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Rnd Classifier  (AUC = %0.5f)' % (ns_auc))
    #plt.plot([0,1], [0,1], 'k--', label="Random Classifier" )
    
    plt.title(g_title)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(prop={'size': 10})
    #plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.savefig('gru_rnn_roc_curve.png')
    plt.show()

def plot_pr_re_curve_CV(precision_list,recall_list,pr_auc_list,no_skill,ns_pr_auc,mean_precision,mean_recall,mean_pr_auc,g_title):
    plt.clf()
    plt.figure()
    for k in range(len(precision_list)):
        #plot fold ROCs
        plt.plot(recall_list[k], precision_list[k], lw=2, alpha=0.3, label='Pr-Re fold %d (AUC = %0.5f)' % (k+1, pr_auc_list[k]))
        
    #plt avg 5-fold CV classifier    
    plt.plot(mean_recall, mean_precision,color='blue', marker='.', label=r'Mean Pr-Re (AUC = %0.5f )' % (mean_pr_auc),lw=2, alpha=1)
    
     #plt rnd classifier
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random Classifier (AUC = %0.5f)' %(ns_pr_auc))

    plt.title(g_title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(prop={'size': 10})
    plt.savefig('gru_rnn_pr_re_curve.png')
    plt.show()

def train_evaluate_model(data_x, data_y, classifier, classifierTypeName, numFeat, num_epochs, num_bsize,valid_dataset):
    strtfdKFold = StratifiedKFold(n_splits=5)
    kfold = strtfdKFold.split(data_x, data_y)
    scores=[]
    predicted_targets = np.array([])
    actual_targets = np.array([])
    pr_re50_list=[]
    pr_list=[]
    re_list=[]
    acc_list=[]
    ce_list=[]
    sp_list=[]
    fpr_list=[]
    fnr_list=[]
    npv_list=[]
    fdr_list=[]
    tprs = []
    aucs = []
    mean_fpr=np.linspace(0,1,100)
    fpr_cv=[]
    tpr_cv=[]
    Y_testa=[]
    Y_proba=[]
    recall_list=[]
    precision_list=[]
    pr_auc_list=[]
    f1_list=[]
    
    train_hist_acc=[]
    test_hist_acc=[]
    train_hist_loss=[]
    test_hist_loss=[]

        
    print(f"Tarining and Testing with 5-CV a {classifierTypeName} with {numFeat} features")
    for k, (train_ix, test_ix) in enumerate(kfold):
        train_x, train_y, test_x, test_y = data_x.iloc[train_ix,:], data_y.iloc[train_ix], data_x.iloc[test_ix,:], data_y.iloc[test_ix]
        
        #Shape data
        #transform data into numpy array 
        x_training_data=np.array(train_x)
        y_training_data=np.array(train_y)
        x_test_data=np.array(test_x)
        y_test_data=np.array(test_y)

        #check data
        # print(x_training_data)
        # print(y_training_data)
        # print(x_test_data)
        # print(y_test_data)

        #Verifying the shape of the NumPy arrays
        # print(x_training_data.shape)
        # print(y_training_data.shape)
        # print(x_test_data.shape)
        # print(y_test_data.shape)


        #Reshaping the NumPy array to meet TensorFlow standards
        x_training_data = np.reshape(x_training_data, (x_training_data.shape[0],  x_training_data.shape[1], 1))
        x_test_data = np.reshape(x_test_data, (x_test_data.shape[0],  x_test_data.shape[1], 1))

        #Printing the new shape 
        # print(x_training_data.shape)
        # print(x_test_data.shape)


        # Fit the classifier
        print(f"Fold {k+1}:____________________________")
        print()
        level0 = list()
        level0.append(('knn', KNeighborsRegressor()))    
        level0.append(('lr', LinearRegression()))
        level0.append(('DTR', tree.DecisionTreeRegressor()))
        # define meta learner model
        level1 = LinearRegression()
        # define the stacking ensemble
        model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
        # fit the model on all available data
        model.fit(x_training_data[:,:,0], y_training_data)
        # make a prediction for one example
        #data = [[0.59332206,-0.56637507,1.34808718,-0.57054047,-0.72480487,1.05648449,0.77744852,0.07361796,0.88398267,2.02843157,1.01902732,0.11227799,0.94218853,0.26741783,0.91458143,-0.72759572,1.08842814,-0.61450942,-0.69387293,1.69169009]]
        #yhat = model.predict(data)
        #print('Predicted Value: %.3f' % (yhat))

        #classifier_history = classifier.fit(x_training_data, y_training_data, epochs = num_epochs, batch_size = num_bsize, callbacks=callbacks, validation_data=valid_dataset,verbose=True)


        # Predict the labels of the test set samples
        predicted_labels_proba = model.predict(x_test_data[:,:,0])
        predicted_labels=tf.greater(predicted_labels_proba, .5)
        
        #Store info for the fold
        predicted_targets = np.append(predicted_targets, predicted_labels)#Y_pred per fold in array
        actual_targets = np.append(actual_targets, test_y) # Y_target or test_y is the target or class labels of the fold
        
        
        #ROC Curve + AUC for the fold
        Y_class_score = model.predict(x_test_data[:,:,0]) #Predicition probabilities flattened to 1D array
        fpr, tpr, thresholds = roc_curve(test_y, Y_class_score) #smoother curve
        tprs.append(interp(mean_fpr,fpr,tpr))
        #roc_auc=auc(fpr,tpr)
        roc_auc= roc_auc_score(test_y,predicted_labels)
        aucs.append(roc_auc)
        fpr_cv.append(fpr)
        tpr_cv.append(tpr)
        
        #Pr-Re Curve + AUC for the fold 
        precision, recall, thresholds = precision_recall_curve(test_y, Y_class_score)
        f1, pr_auc = f1_score(test_y, predicted_labels), auc(recall, precision)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        pr_auc_list.append(pr_auc)
        Y_testa.append(test_y)
        Y_proba.append(Y_class_score)
        
        #Get pr at recall 50 
        i_re50=[index for (index, number) in enumerate(recall) if (number < 0.51 and number >= 0.5)][-1]
        #print(i_re50)
        pr_re50=precision[i_re50] #precision at re 50%
        #print(pr_50)
        pr_re50_list.append(pr_re50)
        
        train_ac=[]
        test_ac=[]
        
        train_ac = model.predict(x_training_data[:,:,0])
        test_ac = model.predict(x_test_data[:,:,0])
        train_acc = accuracy_score(y_training_data, train_ac>0.5)
        test_acc = accuracy_score(y_test_data, test_ac>0.5)
        
        #_, train_acc = classifier.evaluate(x_training_data,y_training_data, verbose=0)
        #_, test_acc = classifier.evaluate(x_test_data, y_test_data, verbose=0)
        print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
        print()
        print("Pr at Re=0.5: %f" % pr_re50)
        print()
       
        #Class Report
        # print(f'Classification Report for {classifierTypeName} features')#of the fold
        # cr=classification_report(test_y, predicted_labels)
        
        #get individual confusion matrix
        print()
        cm=print_confusion_matrix(test_y,predicted_labels)
        #print(cm)
        print()
        cm_stat=compute_cm_stat(cm) #dictionary of
        print(f'F1 : {f1:.5f}')
        print()
        print()
        #collect all stats per fold to have CIs 
        acc_list.append(cm_stat["acc"])
        ce_list.append(cm_stat["ce"])
        pr_list.append(cm_stat["pr"])
        re_list.append(cm_stat["re"])
        sp_list.append(cm_stat["sp"])
        fpr_list.append(cm_stat["fpr"])
        fnr_list.append(cm_stat["fnr"])
        npv_list.append(cm_stat["npv"])
        fdr_list.append(cm_stat["fdr"])
       

    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(test_y))]
    ns_auc = roc_auc_score(test_y, ns_probs)
    ns_fpr, ns_tpr, _ = roc_curve(test_y, ns_probs)
    Y_testac=np.concatenate(Y_testa)
    Y_probac=np.concatenate(Y_proba)
    #generate a mean ROC curve
    mean_tpr=np.mean(tprs,axis=0)
    mean_auc=auc(mean_fpr,mean_tpr)
    
    #send data to plot the ROC figure 
    g_title="ROC curve for " + classifierTypeName + " with " + numFeat + " features"
    
    mean_fpr, mean_tpr, thresholds = roc_curve(Y_testac,Y_probac) #smoother fuller curve
    plot_roc_CV(fpr_cv,tpr_cv,aucs, ns_fpr,ns_tpr,ns_auc,mean_tpr,mean_fpr,mean_auc,g_title)
   
    print(f'5-CV ROC AUC for {classifierTypeName} with {numFeat} features: {mean_auc:.5f}')
    roc_plt_data={"mean_tpr":mean_tpr,"mean_fpr":mean_fpr,"mean_auc":mean_auc,
                  "ns_fpr":ns_fpr, "ns_tpr":ns_tpr,"ns_auc":ns_auc}
    
    #send data to plot the Pr-Re Curve figure 
    no_skill = len(y_test_data[y_test_data==1]) / len(y_test_data)
    ns_pr_auc=no_skill
    
    mean_precision,mean_recall,_=precision_recall_curve(Y_testac,Y_probac)
    mean_pr_auc=auc(mean_recall,mean_precision)
    g_title2="Pr-Re curve for " + classifierTypeName + " with " + numFeat + " features"
    plot_pr_re_curve_CV(precision_list,recall_list,pr_auc_list,no_skill,ns_pr_auc,mean_precision,mean_recall,mean_pr_auc,g_title2)
    
    print(f'5-CV Pr-Re AUC for {classifierTypeName} with {numFeat} features: {mean_pr_auc:.5f}')
    prc_plt_data={"no_skill":no_skill,"ns_pr_auc":ns_pr_auc,
                  "mean_pr":mean_precision,"mean_re":mean_recall, "mean_pr_auc":mean_pr_auc}
    
   

    print()
    
    print(f'5-CV Performance Measures with CIs for {classifierTypeName} with {numFeat} features')
    pr_mean=np.mean(pr_list)
    pr_std=np.std(pr_list)
    
    pr_re50_mean=np.mean(pr_re50_list)
    pr_re50_std=np.std(pr_re50_list)
    
    re_mean=np.mean(re_list)
    re_std=np.std(re_list)
    
    acc_mean=np.mean(acc_list)
    acc_std=np.std(acc_list)
    
    ce_mean=np.mean(ce_list)
    ce_std=np.std(ce_list)
    
    sp_mean=np.mean(sp_list)
    sp_std=np.std(sp_list)
    
    fpr_mean=np.mean(fpr_list)
    fpr_std=np.std(fpr_list)
    
    fnr_mean=np.mean(fnr_list)
    fnr_std=np.std(fnr_list)
    
    npv_mean=np.mean(npv_list)
    npv_std=np.std(npv_list)
    
    fdr_mean=np.mean(fdr_list)
    fdr_std=np.std(fdr_list)
    
    f1_mean=np.mean(f1_list)
    f1_std=np.std(f1_list)
    
    cm_CI_stats={"acc":acc_mean,"acc_std":acc_std,"pr":pr_mean,"pr_std":pr_std,"re":re_mean,"re_std":re_std,
                "sp":sp_mean,"sp_std":sp_std,"ce":ce_mean,"ce_std":ce_std,"fpr":fpr_mean,"fpr_std":fpr_std,
                "fnr":fnr_mean,"fnr_std":fnr_std,"npv":npv_mean,"npv_std":npv_std, "fdr":fdr_mean,"fdr_std":fdr_std,
                "f1":f1_mean,"f1_std":f1_std, "pr50":pr_re50_mean, "pr50_std":pr_re50_std}
    
    print()
    print('5-CV Pr at Re 0.50: %.5f +/- %.12f' %(pr_re50_mean, pr_re50_std))
    print()
    print('5-CV Accuracy: %.5f +/- %.12f' %(acc_mean, acc_std))
    print('5-CV Pr: %.5f +/- %.12f' %(pr_mean, pr_std))
    print('5-CV Re/Sn/TPR: %.5f +/- %.12f' %(re_mean, re_std))
    print('5-CV Sp: %.5f +/- %.12f' %(sp_mean, sp_std))
    print('5-CV FPR: %.5f +/- %.12f' %(fpr_mean, fpr_std))
    print('5-CV FNR: %.5f +/- %.12f' %(fnr_mean, fnr_std))
    print('5-CV NPV: %.5f +/- %.12f' %(npv_mean, npv_std))
    print('5-CV FDR: %.5f +/- %.12f' %(fdr_mean, fdr_std))
    print('5-CV F1-Score: %.5f +/- %.12f' %(f1_mean, f1_std))
    print()
    print(f'5-CV Concatenated Confusion Matrix for {classifierTypeName} with {numFeat} features')
    cm=print_confusion_matrix(actual_targets,predicted_targets) #concatenated
    #Print Accuracy, Pr, Re, Sp
    print()
    print(f'5-CV Concatenated Confusion Matrix Perform. Measures')
    print()
    cm_stat=compute_cm_stat(cm) #dictionary of
    


    #plot training history -loss
    plt.clf()
    plt.title(f"{classifierTypeName} Loss")
     
    plt.plot(train_hist_loss, label='train')
    plt.plot(test_hist_loss, label='test')
    plt.legend()
    plt.savefig('gru_rnn_training_history_loss.png')
    plt.show()

    #plot training history - acc
    plt.clf()
    plt.title(f"{classifierTypeName} Accuracy")
    plt.plot(train_hist_acc, label='train')
    plt.plot(test_hist_acc, label='test')
    plt.legend()
    plt.savefig('gru_rnn_training_history_acc.png')
    plt.show()

    
    return {"cm":cm, "cm_stat": cm_stat, "auc_roc":mean_auc, "auc_prc":mean_pr_auc,
             "cm_CI_stats":cm_CI_stats, "roc_plt":roc_plt_data, "prc_plt":prc_plt_data, 
             "model_trained":model}




def train_evaluate_model_meta(data_x, data_y, classifier, classifierTypeName, numFeat, num_epochs, num_bsize,valid_dataset,model_GRU, model_LSTM):
    strtfdKFold = StratifiedKFold(n_splits=5)
    kfold = strtfdKFold.split(data_x, data_y)
    scores=[]
    predicted_targets = np.array([])
    actual_targets = np.array([])
    pr_re50_list=[]
    pr_list=[]
    re_list=[]
    acc_list=[]
    ce_list=[]
    sp_list=[]
    fpr_list=[]
    fnr_list=[]
    npv_list=[]
    fdr_list=[]
    tprs = []
    aucs = []
    mean_fpr=np.linspace(0,1,100)
    fpr_cv=[]
    tpr_cv=[]
    Y_testa=[]
    Y_proba=[]
    recall_list=[]
    precision_list=[]
    pr_auc_list=[]
    f1_list=[]
    
    train_hist_acc=[]
    test_hist_acc=[]
    train_hist_loss=[]
    test_hist_loss=[]

        
    print(f"Tarining and Testing with 5-CV a {classifierTypeName} with {numFeat} features")
    for k, (train_ix, test_ix) in enumerate(kfold):
        train_x, train_y, test_x, test_y = data_x.iloc[train_ix,:], data_y.iloc[train_ix], data_x.iloc[test_ix,:], data_y.iloc[test_ix]
        
        #Shape data
        #transform data into numpy array 
        x_training_data=np.array(train_x)
        y_training_data=np.array(train_y)
        x_test_data=np.array(test_x)
        y_test_data=np.array(test_y)

        #check data
        # print(x_training_data)
        # print(y_training_data)
        # print(x_test_data)
        # print(y_test_data)

        #Verifying the shape of the NumPy arrays
        # print(x_training_data.shape)
        # print(y_training_data.shape)
        # print(x_test_data.shape)
        # print(y_test_data.shape)


        #Reshaping the NumPy array to meet TensorFlow standards
        x_training_data = np.reshape(x_training_data, (x_training_data.shape[0],  x_training_data.shape[1], 1))
        x_test_data = np.reshape(x_test_data, (x_test_data.shape[0],  x_test_data.shape[1], 1))

        #Printing the new shape 
        print('x_training_data.shape===================',x_training_data.shape)
        # print(x_test_data.shape)
        
        yhat_gru = []
        yhat_lstm = []
        
        yhat_gru.append(model_GRU.predict((x_training_data), verbose=0))
        yhat_lstm.append(model_LSTM.predict(x_training_data, verbose=0))
        yhat_gru = np.array(yhat_gru)
        yhat_lstm = np.array(yhat_lstm)
        yhat_gru = np.array(yhat_gru)
        yhat_lstm = np.array(yhat_lstm)
        tempp = np.array((yhat_gru,yhat_lstm))
        tempp=np.squeeze(tempp)

        #temp = np.concatenate((tempp[:,:,None],x_training_data))
        temp = np.zeros((65687,7,1))
        temp[:,0:5,:] = x_training_data
        temp[:,5,0] = yhat_gru[0,:,0]
        temp[:,6,0] = yhat_lstm[0,:,0]
        print('temp.shape========',temp.shape)


        # Fit the classifier
        print(f"Fold {k+1}:____________________________")
        print()
        classifier_history = classifier.fit(temp, y_training_data, epochs = num_epochs, batch_size = num_bsize, callbacks=callbacks, validation_data=valid_dataset,verbose=True)

        #store claassifier acc 
        train_hist_acc=np.append(train_hist_acc, classifier_history.history['acc'])
        test_hist_acc=np.append(test_hist_acc, classifier_history.history['val_acc'])
        train_hist_loss=np.append(train_hist_loss, classifier_history.history['loss'])
        test_hist_loss=np.append(test_hist_loss, classifier_history.history['val_loss'])

        # Predict the labels of the test set samples
        predicted_labels_proba = classifier.predict(x_test_data)
        predicted_labels=tf.greater(predicted_labels_proba, .5)
        
        #Store info for the fold
        predicted_targets = np.append(predicted_targets, predicted_labels)#Y_pred per fold in array
        actual_targets = np.append(actual_targets, test_y) # Y_target or test_y is the target or class labels of the fold
        
        
        #ROC Curve + AUC for the fold
        Y_class_score = classifier.predict(x_test_data).ravel() #Predicition probabilities flattened to 1D array
        fpr, tpr, thresholds = roc_curve(test_y, Y_class_score) #smoother curve
        tprs.append(interp(mean_fpr,fpr,tpr))
        #roc_auc=auc(fpr,tpr)
        roc_auc= roc_auc_score(test_y,predicted_labels)
        aucs.append(roc_auc)
        fpr_cv.append(fpr)
        tpr_cv.append(tpr)
        
        #Pr-Re Curve + AUC for the fold 
        precision, recall, thresholds = precision_recall_curve(test_y, Y_class_score)
        f1, pr_auc = f1_score(test_y, predicted_labels), auc(recall, precision)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        pr_auc_list.append(pr_auc)
        Y_testa.append(test_y)
        Y_proba.append(Y_class_score)
        
        #Get pr at recall 50 
        i_re50=[index for (index, number) in enumerate(recall) if (number < 0.51 and number >= 0.5)][-1]
        #print(i_re50)
        pr_re50=precision[i_re50] #precision at re 50%
        #print(pr_50)
        pr_re50_list.append(pr_re50)
        
        
        print()
        
        _, train_acc = classifier.evaluate(x_training_data,y_training_data, verbose=0)
        _, test_acc = classifier.evaluate(x_test_data, y_test_data, verbose=0)
        print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
        print()
        print("Pr at Re=0.5: %f" % pr_re50)
        print()
       
        #Class Report
        # print(f'Classification Report for {classifierTypeName} features')#of the fold
        # cr=classification_report(test_y, predicted_labels)
        
        #get individual confusion matrix
        print()
        cm=print_confusion_matrix(test_y,predicted_labels)
        #print(cm)
        print()
        cm_stat=compute_cm_stat(cm) #dictionary of
        print(f'F1 : {f1:.5f}')
        print()
        print()
        #collect all stats per fold to have CIs 
        acc_list.append(cm_stat["acc"])
        ce_list.append(cm_stat["ce"])
        pr_list.append(cm_stat["pr"])
        re_list.append(cm_stat["re"])
        sp_list.append(cm_stat["sp"])
        fpr_list.append(cm_stat["fpr"])
        fnr_list.append(cm_stat["fnr"])
        npv_list.append(cm_stat["npv"])
        fdr_list.append(cm_stat["fdr"])
       

    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(test_y))]
    ns_auc = roc_auc_score(test_y, ns_probs)
    ns_fpr, ns_tpr, _ = roc_curve(test_y, ns_probs)
    Y_testac=np.concatenate(Y_testa)
    Y_probac=np.concatenate(Y_proba)
    #generate a mean ROC curve
    mean_tpr=np.mean(tprs,axis=0)
    mean_auc=auc(mean_fpr,mean_tpr)
    
    #send data to plot the ROC figure 
    g_title="ROC curve for " + classifierTypeName + " with " + numFeat + " features"
    
    mean_fpr, mean_tpr, thresholds = roc_curve(Y_testac,Y_probac) #smoother fuller curve
    plot_roc_CV(fpr_cv,tpr_cv,aucs, ns_fpr,ns_tpr,ns_auc,mean_tpr,mean_fpr,mean_auc,g_title)
   
    print(f'5-CV ROC AUC for {classifierTypeName} with {numFeat} features: {mean_auc:.5f}')
    roc_plt_data={"mean_tpr":mean_tpr,"mean_fpr":mean_fpr,"mean_auc":mean_auc,
                  "ns_fpr":ns_fpr, "ns_tpr":ns_tpr,"ns_auc":ns_auc}
    
    #send data to plot the Pr-Re Curve figure 
    no_skill = len(y_test_data[y_test_data==1]) / len(y_test_data)
    ns_pr_auc=no_skill
    
    mean_precision,mean_recall,_=precision_recall_curve(Y_testac,Y_probac)
    mean_pr_auc=auc(mean_recall,mean_precision)
    g_title2="Pr-Re curve for " + classifierTypeName + " with " + numFeat + " features"
    plot_pr_re_curve_CV(precision_list,recall_list,pr_auc_list,no_skill,ns_pr_auc,mean_precision,mean_recall,mean_pr_auc,g_title2)
    
    print(f'5-CV Pr-Re AUC for {classifierTypeName} with {numFeat} features: {mean_pr_auc:.5f}')
    prc_plt_data={"no_skill":no_skill,"ns_pr_auc":ns_pr_auc,
                  "mean_pr":mean_precision,"mean_re":mean_recall, "mean_pr_auc":mean_pr_auc}
    
   

    print()
    
    print(f'5-CV Performance Measures with CIs for {classifierTypeName} with {numFeat} features')
    pr_mean=np.mean(pr_list)
    pr_std=np.std(pr_list)
    
    pr_re50_mean=np.mean(pr_re50_list)
    pr_re50_std=np.std(pr_re50_list)
    
    re_mean=np.mean(re_list)
    re_std=np.std(re_list)
    
    acc_mean=np.mean(acc_list)
    acc_std=np.std(acc_list)
    
    ce_mean=np.mean(ce_list)
    ce_std=np.std(ce_list)
    
    sp_mean=np.mean(sp_list)
    sp_std=np.std(sp_list)
    
    fpr_mean=np.mean(fpr_list)
    fpr_std=np.std(fpr_list)
    
    fnr_mean=np.mean(fnr_list)
    fnr_std=np.std(fnr_list)
    
    npv_mean=np.mean(npv_list)
    npv_std=np.std(npv_list)
    
    fdr_mean=np.mean(fdr_list)
    fdr_std=np.std(fdr_list)
    
    f1_mean=np.mean(f1_list)
    f1_std=np.std(f1_list)
    
    cm_CI_stats={"acc":acc_mean,"acc_std":acc_std,"pr":pr_mean,"pr_std":pr_std,"re":re_mean,"re_std":re_std,
                "sp":sp_mean,"sp_std":sp_std,"ce":ce_mean,"ce_std":ce_std,"fpr":fpr_mean,"fpr_std":fpr_std,
                "fnr":fnr_mean,"fnr_std":fnr_std,"npv":npv_mean,"npv_std":npv_std, "fdr":fdr_mean,"fdr_std":fdr_std,
                "f1":f1_mean,"f1_std":f1_std, "pr50":pr_re50_mean, "pr50_std":pr_re50_std}
    
    print()
    print('5-CV Pr at Re 0.50: %.5f +/- %.12f' %(pr_re50_mean, pr_re50_std))
    print()
    print('5-CV Accuracy: %.5f +/- %.12f' %(acc_mean, acc_std))
    print('5-CV Pr: %.5f +/- %.12f' %(pr_mean, pr_std))
    print('5-CV Re/Sn/TPR: %.5f +/- %.12f' %(re_mean, re_std))
    print('5-CV Sp: %.5f +/- %.12f' %(sp_mean, sp_std))
    print('5-CV FPR: %.5f +/- %.12f' %(fpr_mean, fpr_std))
    print('5-CV FNR: %.5f +/- %.12f' %(fnr_mean, fnr_std))
    print('5-CV NPV: %.5f +/- %.12f' %(npv_mean, npv_std))
    print('5-CV FDR: %.5f +/- %.12f' %(fdr_mean, fdr_std))
    print('5-CV F1-Score: %.5f +/- %.12f' %(f1_mean, f1_std))
    print()
    print(f'5-CV Concatenated Confusion Matrix for {classifierTypeName} with {numFeat} features')
    cm=print_confusion_matrix(actual_targets,predicted_targets) #concatenated
    #Print Accuracy, Pr, Re, Sp
    print()
    print(f'5-CV Concatenated Confusion Matrix Perform. Measures')
    print()
    cm_stat=compute_cm_stat(cm) #dictionary of
    


    #plot training history -loss
    plt.clf()
    plt.title(f"{classifierTypeName} Loss")
     
    plt.plot(train_hist_loss, label='train')
    plt.plot(test_hist_loss, label='test')
    plt.legend()
    plt.savefig('gru_rnn_training_history_loss.png')
    plt.show()

    #plot training history - acc
    plt.clf()
    plt.title(f"{classifierTypeName} Accuracy")
    plt.plot(train_hist_acc, label='train')
    plt.plot(test_hist_acc, label='test')
    plt.legend()
    plt.savefig('gru_rnn_training_history_acc.png')
    plt.show()

    
    return {"cm":cm, "cm_stat": cm_stat, "auc_roc":mean_auc, "auc_prc":mean_pr_auc,
             "cm_CI_stats":cm_CI_stats, "roc_plt":roc_plt_data, "prc_plt":prc_plt_data, 
             "model_trained":classifier}







def test_dataset_evaluate(classifier, x_test_data, y_test_data, classifierTypeName, numFeats):
    #Evaluate on held out training set 
    
    
    
    # evaluate the model
    _, test_acc = classifier.evaluate(x_test_data, y_test_data, verbose=0)
    
    #Generating our predicted values probabilities
    rnn_preda = classifier.predict(x_test_data).ravel()
    #print(rnn_preda)

    #labels 
    rnn_rawpreda = classifier.predict(x_test_data)
    rnn_predClass=tf.greater(rnn_rawpreda, .5)  #sigmoid was used that is why NOTE!!!!
    #print(rnn_predClass)
    conf=confusion_matrix(y_test_data, rnn_predClass)
    print(conf)
    print_confusion_matrix(y_test_data, rnn_predClass)
    plt.clf()
    
    fpr, tpr, thresholds = roc_curve(y_test_data, rnn_preda) #smoother curve
    plt.figure(figsize=(6,4))
    ROC_AUC = roc_auc_score(Y_test, rnn_preda)
    plt.plot(fpr, tpr, linewidth=2, label='ROC %s %s feats. (AUC = %0.5f)' % (classifierTypeName, numFeats, ROC_AUC))
    

    
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test_data))]
    ns_auc = roc_auc_score(y_test_data, ns_probs)
    ns_fpr, ns_tpr, _ = roc_curve(y_test_data, ns_probs)
    plt.plot(ns_fpr, ns_tpr, linestyle='--',label="Random Classifier" )
    #plt.plot([0,1], [0,1], 'k--', label="Random Classifier" )
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve for held out 25 percent test data')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend()
    plt.savefig('gru_rnn_roc_held_25_percent.png')
    plt.show()
    
    # calculate precision-recall curve
    pr, re, thresholds = precision_recall_curve(y_test_data, rnn_preda)
    pr_auc = auc(re, pr)
    # summarize scores
   
    # plot the precision-recall curves
   
    no_skill = len(y_test_data[y_test_data==1]) / len(y_test_data)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random Classifier')
    plt.plot(re, pr, marker='.', label='Pr-Re %s %s feats. (AUC = %0.5f)' % (classifierTypeName, numFeats, pr_auc))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Pr-Re curve for held out 25 percent test data')
    plt.legend()
    plt.savefig('gru_rnn_pr_re_held_25_percent.png')
    plt.show()
    
    # print(pr)
    # print(re)
    #Get pr at recall 50 
    i_re50=[index for (index, number) in enumerate(re) if (number < 0.51 and number >= 0.5)][-1]
    
    #print(i_re50)
    pr_50=pr[i_re50] #precision at re 50%
    #print(pr_50)
    print()
    print("Pr at Re=0.5: %f" % pr_50)
    print()
    
    print('RNN AUC ROC : {:.4f}'.format(ROC_AUC))
    print('Rnd.AUC : %.4f' % (ns_auc))
    print()
    print('RNN AUC Pr-Re: auc=%.3f' % (pr_auc))
    print('Rnd. AUC Pr-Re: {:.4f}'.format(no_skill))
    
    print('Test with held out 25 percent test data %f' % (test_acc))
    rnn_accuracy=accuracy_score(y_test_data, rnn_predClass)
    print("Accuracy: %f" % rnn_accuracy)
    rnn_Re=recall_score(y_test_data, rnn_predClass)
    print("Re %f" % rnn_Re)
    rnn_Pr=precision_score(y_test_data, rnn_predClass)
    print("Pr: %f" % rnn_Pr)
    rnn_F1=f1_score(y_test_data, rnn_predClass)
    print("F1: %f" % rnn_F1)
    
    #Plotting our predicted values for regression

    # plt.clf() #This clears the old plot from our canvas

    # plt.plot(rnn_preda)

    #Plotting the predicted values against  actual 

    # plt.plot(rnn_preda, color = '#135485', label = "Predictions")

    # plt.plot(y_test_data, color = 'black', label = "Real Data")

    # plt.title('DTI Predictions')
   



#READ TRAINING DATA SET PROVIDED

df = pd.read_csv ('train_data.csv')#header=None)
print(df)
print(df.head())

X_feat=df.iloc[:,0:336] #separate into feature and target data - removed KIBA and target
Y_target=df.iloc[:,-1:] #binary labels

print(X_feat)
print(Y_target)

#PRE_PROCESSING WITH MINMAX Scaler
X=pd.DataFrame(X_feat)
scaler = MinMaxScaler()
X_feat_t=scaler.fit_transform(X)
X_feat_s=pd.DataFrame(X_feat_t) #Scaled features
print(X_feat_s)

#encode Target labels 0 for neg/False 1 for pos/True 
from sklearn.preprocessing import LabelEncoder
random_state = 123
y=Y_target
labEnc = LabelEncoder()
y = labEnc.fit_transform(y)



#ANOVA F
# feature extraction
test2 = SelectKBest(score_func=f_classif, k=5) # 5 best features 
fit2 = test2.fit(X_feat_s, Y_target)
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
plt.savefig('gru_rnn_anova_top_25.png')
plt.show()
plt.clf()



#Pre-processed or scaled and selected 5 features
X_s_t=pd.DataFrame(X_feat_new2)

#Pre-pocessed + feature slected data 
data_x=X_s_t
data_y=y

#split into train test - hold out
X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.25, stratify=data_y,random_state=123) #stratified

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


#Initializing our recurrent neural network
ANN = Sequential()

ANN.add(Dense(12, input_shape=(x_training_data.shape[1]+2,1), activation='relu'))
ANN.add(Dropout(0.2))
ANN.add(Dense(8, activation='relu'))
ANN.add(Dropout(0.2))
ANN.add(Dense(1, activation='sigmoid'))
ANN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['acc'])

#Compiling the recurrent neural network

# Create callbacks: Early Stopping Pt and check point to later reload the trained classifier
callbacks=[]
es=EarlyStopping(monitor='val_loss', patience=50)
callbacks.append(es)
mcp=ModelCheckpoint('../gru_rnn_models/ANN.h5', save_best_only=True, save_weights_only=False)
callbacks.append(mcp)



#Training the recurrent neural network
classifierTypeName="Ensemble of models"
numFeat="5"
num_epochs=100 #per fold - 5 folds - so 500
num_bsize=32
valid_dataset=(x_test_data,y_test_data)

dfX_train= pd.DataFrame(X_train)
dfY_train= pd.DataFrame(Y_train)

train_ANN=train_evaluate_model(dfX_train, dfY_train, ANN, classifierTypeName, numFeat, num_epochs, num_bsize, valid_dataset)

model_trained=train_ANN["model_trained"]

# save model and architecture to single file
model_trained.save("ANN.h5")
print("Saved model to disk")

test_dataset_evaluate(model_trained, x_test_data, y_test_data, classifierTypeName, numFeat) 


#WHEN PROF WILL give us the data the required probailities are computed using:

#1. Reload saved trained model: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
#2. Preprocess using MinMaxScaler
#3. Extract features using anova_column_name list 
#4. Generating our predicted values with probabilities using trained model
# rnn_preda = model_trained.predict(x_test_data).ravel()
# print(rnn_preda)
'''
        yhat_gru = []
        yhat_lstm = []
        
        for i in x_training_data:
            print('i.shape============= ',i.shape)
            #yhat_gru.append(model_GRU.predict(i, verbose=0))
            yhat_lstm.append(model_LSTM.predict(i, verbose=0))
            print('yhat_gru.shape============= ',yhat_gru.shape)
'''

