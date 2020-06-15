import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif,SelectKBest
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score



#Initialising

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
os.chdir('/Users/praveenthenraj/Desktop/PycharmProjects/Trial/Hackerearth/Novartis/Data/Dataset')

#Reading the data

train=pd.read_csv('Train.csv')
test=pd.read_csv('Test.csv')
sample=pd.read_csv('sample_submission.csv')
submis=pd.DataFrame()


#Initialising Variables and Data Structures

imp_feature=[]



#Basic Health Checks

print('Training Data Shape : '+str(train.shape)+'\n')                                                           #(23856, 18)
print('Testing Data Shape  : '+str(test.shape)+'\n')                                                            #(15903, 17)
print('Training Data Non-int type : '+'\n'+'\n'+str(train.dtypes!=np.int64)+'\n')                               #INCIDENT_ID,DATE,X_12
print('Testing Data Non-int type : '+'\n'+'\n'+str(test.dtypes!=np.int64)+'\n')                                 #INCIDENT_ID,DATE,X_12
print('Training Data Null Columns : '+'\n'+'\n'+str(train.isna().any())+'\n')                                   #X_12
print('Testing Data Null Columns : '+'\n'+'\n'+str(test.isna().any())+'\n')                                     #X_12
print('Count of Null Data in Train : '+'\n'+'\n'+str((train[train.X_12.isnull()]=='True').count())+'\n')        #182
print('Count of Null Data in Test : '+'\n'+'\n'+str((test[test.X_12.isnull()]=='True').count())+'\n')           #182
print('Balance of Dependant Variables : '+'\n'+'\n'+str(train['MULTIPLE_OFFENSE'].value_counts())+'\n')         #0-1068 1-22788
print('No of duplicates in Train Data : '+str(len(train.loc[train['INCIDENT_ID'].duplicated()]==True))+'\n')
print('No of duplicates in Train Data : '+str(len(test.loc[test['INCIDENT_ID'].duplicated()]==True))+'\n')



#EDA

# Plot 1: To identify the distribution of values in X_12 to impute for NaN values in X_12

train['X_12'].value_counts().plot(kind='bar')
plt.xlabel('X_12')
plt.ylabel('Value_counts')
plt.title('Distribution of values in X_12')
plt.show()

# Plot 2: To identify the distribution of each predictor variable and their respective variance

train_eda = train.iloc[:, 2:17]

for i in train_eda.columns:
    std = np.round(np.std(train[i]), 2)
    mean = np.round(np.mean(train[i]), 2)
    train[i].value_counts().plot(kind='bar',label=r'Mean    ' + ' - ' + str(mean) + '\n' + r'Std Dev' + ' - ' + str(std))
    plt.xlabel(i)
    plt.ylabel('Value_counts')
    plt.legend()
    plt.title('Distribution of Anonymized Logging parameter ' + i)
    plt.show()


# Plot 3: To identify the correlation between the various dependant variables

sns.heatmap(train.corr(), annot=True)
plt.show()



#Feature Preprocessing and Feature Transformation:

#Filling the na values with 1 as the EDA suggested that X_12 has maximum of 1

def feature_preprocess_transformation(data):
    global train,test
    data['X_12']=data['X_12'].fillna(value=1.0).astype(int)                                                    #Since from the above EDA, we can see more than 75 percentile of X_12 is 1


#Calling the preprocessing function for Train and Test Data

feature_preprocess_transformation(train)
feature_preprocess_transformation(test)



#Feature Importance and Train_Test Data Split

indep=train.drop(['INCIDENT_ID','DATE','MULTIPLE_OFFENSE'],axis=1).values
dep=train['MULTIPLE_OFFENSE'].values

x=SelectKBest(score_func=f_classif,k='all').fit(indep,dep)                                                     #Annova Test selected since indep variable is numerical whereas dep variable is categorical
fs=x.transform(indep)
for i in range(len(x.scores_)):
    print('Feature %d: %f' % (i, x.scores_[i]))
    imp_feature.append(i)


#Selecting only features having annova score of more than 1

print([i for i in range(len(x.scores_)) if x.scores_[i]>5])


#Dropping features having Annova score less than 5 and also features that are correlated based on the EDA done earlier

#'X_1','X_4','X_5','X_6','X_7','X_9','X_13','X_14' --- dropped due to low ANNOVA score
#'X_3','X_12'  --- dropped due to interaction effect between these variables and X_2,X_10 respectively


X=train.drop(['INCIDENT_ID','DATE','X_1','X_2','X_4','X_5','X_6','X_7','X_9','X_12','X_13','X_14','MULTIPLE_OFFENSE'],axis=1).values           #Dropping features having Annova score less than 1. X_3 dropped even score is 17 as its score is same as X_2
y=train['MULTIPLE_OFFENSE'].values

#Upsampling the minority class as an imbalanced dataset is observed during the Basic Health checks

sm=SMOTE(sampling_strategy='minority',random_state=2)
X_train_sm,y_train_sm=sm.fit_sample(X,y.ravel())


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y,random_state=20)



#Hyperparameter Tuning

#l=[0.001,0.005,0.01,0.05,0.5,0.1]
#n=[i for i in range(100,300,100)]
#md=[i for i in np.arange(3,8,1)]
#rl=[i for i in np.arange(0,1,0.1)]

#param={'learning_rate':l}
#param={'n_estimators':n,'max_depth':md}
#param={'reg_lambda':rl}
#xgb=XGBClassifier()
#xgb=GridSearchCV(xgb1,param_grid=param,cv=5)
#print(xgb.best_params_)
#print(xgb.best_estimator_)
#y_pred=xgb.best_estimator_.predict(X_test)


# Model
#Based on the above hyperparameter tuning the below parameters have found to be precise

xgb=XGBClassifier(learning_rate=0.7200000000000001,max_depth=3,n_estimators=100,reg_lambda=0.1)
xgb.fit(X_train_sm,y_train_sm)

y_pred=xgb.predict(X_test)

tn,fp,fn,tp=confusion_matrix(y_test,y_pred).ravel()
recall=recall_score(y_test,y_pred)
print(recall)
print(tn,fp,fn,tp)


#Processing the test file

X_sub=test.drop(['INCIDENT_ID','DATE','X_1','X_2','X_4','X_5','X_6','X_7','X_9','X_12','X_13','X_14'],axis=1).values                           #Dropping features having Annova score less than 1. X_3 dropped even score is 17 as its score is same as X_2
y_sub=xgb.predict(X_sub)


#Preparing the submission file

submis['INCIDENT_ID']=test['INCIDENT_ID']
submis['MULTIPLE_OFFENSE']=y_sub
submis.to_csv('outfile.csv',index=False)
