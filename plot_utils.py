# IMPORTS
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import f1_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# CUSTOM FUNCTIONS FOR METRICS
def print_metrics(y_true, preds, model_name=None):
    '''
    INPUT:
    y_true - the y values that are actually true in the dataset (numpy array or pandas series)
    preds - the predictions for those values from some model (numpy array or pandas series)
    model_name - (str - optional) a name associated with the model if you would like to add it to the print statements 
    
    OUTPUT:
    None - prints the accuracy, precision, recall, and F1 score
    '''
    if model_name == None:
        print('Accuracy score: ', format(accuracy_score(y_true, preds)))
        print('Precision score: ', format(precision_score(y_true, preds)))
        print('Recall score: ', format(recall_score(y_true, preds)))
        print('F1 score: ', format(f1_score(y_true, preds)))
        print('\n\n')
    
    else:
        print('Accuracy score for ' + model_name + ' :' , format(accuracy_score(y_true, preds)))
        print('Precision score ' + model_name + ' :', format(precision_score(y_true, preds)))
        print('Recall score ' + model_name + ' :', format(recall_score(y_true, preds)))
        print('F1 score ' + model_name + ' :', format(f1_score(y_true, preds)))
        print('\n\n')
        
        
    
def build_roc_auc(model, X_train, X_test, y_train, y_test):
    '''
    INPUT:
    stuff 
    OUTPUT:
    auc - returns auc as a float
    prints the roc curve
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    from scipy import interp
    
    y_preds = model.fit(X_train, y_train).predict_proba(X_test)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(y_test)):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_preds[:, 1])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_preds[:, 1].ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.show()
    
    return roc_auc_score(y_test, np.round(y_preds[:, 1]))



def missing_data(data):
    '''
    INPUT: dataframe
    OUTPUT: dataframe with total number and percentageof missing values
    '''
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending=False)
    
    return pd.concat([total, percent], axis=1, keys=['total', 'percent'])



# CUSTOM FUNCTIONS FOR PLOTS
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions,'b--', label="Precision")
    plt.plot(thresholds, recalls,'r-', label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc=opt)
    plt.ylim([0,1])


    
def plot_categorical(data, col, size=[8 ,4], xlabel_angle=0, title=''):
    '''use this for ploting the count of categorical features'''
    plotdata = data[col].value_counts()
    plt.figure(figsize = size)
    sns.barplot(x = plotdata.index, y=plotdata.values)
    plt.title(title)
    if xlabel_angle!=0: 
        plt.xticks(rotation=xlabel_angle)
    plt.show()
    


def plot_categorical(data, col, size=[8 ,4], xlabel_angle=0, title=''):
    '''
    use this function for ploting histogram, i.e., the count of categorical features
    '''
    plotdata = data[col].value_counts()
    plt.figure(figsize = size)
    sns.barplot(x = plotdata.index, y=plotdata.values)
    plt.title(title)
    if xlabel_angle!=0: 
        plt.xticks(rotation=xlabel_angle)
    plt.show()
    
    
    
def plot_numerical(data, col, size=[8, 4], bins=50):
    '''
    use this function for ploting the distribution of numercial features
    '''
    plt.figure(figsize=size)
    plt.title("Distribution of %s" % col)
    sns.distplot(data[col].dropna(), kde=True, bins=bins)
    plt.show()
    
    
    
def plot_categorical_bylabel(data, col, size=[12 ,6], xlabel_angle=0, title=''):
    '''
    use it to compare the distribution between label 1 and label 0
    '''
    plt.figure(figsize = size)
    l1 = data.loc[data.TARGET==1, col].value_counts()
    l0 = data.loc[data.TARGET==0, col].value_counts()
    
    plt.subplot(1,2,1)
    sns.barplot(x = l0.index, y=l0.values)
    plt.title('Non-default (Y=0): '+title)
    plt.xticks(rotation=xlabel_angle)
    
    plt.subplot(1,2,2)
    sns.barplot(x = l1.index, y=l1.values)
    plt.title('Default (Y=1): '+title)
    plt.xticks(rotation=xlabel_angle)
    plt.show()
    
    
    
def plot_numerical_bylabel(data, col, size=[8, 4], bins=50):
    '''
    use this function to compare the distribution of numercial features
    '''
    plt.figure(figsize=[12, 6])
    l1 = data.loc[data.TARGET==1, col]
    l0 = data.loc[data.TARGET==0, col]
    
    plt.subplot(1,2,1)
    sns.distplot(l0.dropna(), kde=True,bins=bins)
    plt.title('Non-default (Y=0): Distribution of %s' % col)
    
    plt.subplot(1,2,2)
    sns.distplot(l1.dropna(), kde=True,bins=bins)
    plt.title('Default (Y=1): Distribution of %s' % col)
    
    plt.show()    

