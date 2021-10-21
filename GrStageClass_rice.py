
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 22:00:35 2020

@author: Prakash
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import numpy as np

class Dataset: 
    """
    Dataset class used to store the data, do feature scaling and spliting into
    train and test sets.

    .......

    Attributes
    ----------
    dataset : DataFrame
        A Data Frame that stores the imported dataset.
    target_col : str
        The name of the target column, the column we are trying to predict.
    train_col : list
        The list of all the columns in the dataset.
    id_col : str
        The name of the column used as index, Plot_ID
    rand_seed : int
        The integer value used to set as seed for random number generator to 
        replicate results.
    test_ids  : list
        The list of all the Plot_IDs that are included in the test set.
    train_set  :  DataFrame
        The DataFrame contains all the instances of PlotIDs not included in the
        test set. (The dataframe is however unbalanced)
    test_set  :  DataFrame
        The DataFrame contains all the instances of PlotIDs included in the
        test set via test_ids. (The dataframe is unbalanced)
    x_train  : Array of float 64
        The array contains the balanced and feature scaled(if preprocessing = True)
        training set to be used to fit the models.
    y_train  : Series
        The series contains target values for corresponding features in x_train.
    x_test  : Array of float 64
        The array contains the feature scaled(if preprocessing = True) testing
        set to be used to test the models.
    y_train  : Series
        The series contains target values for corresponding features in x_test.

    Methods
    -------
    create_train_test(num_test_ids, preprocessing)
        Prints the animals name and what sound it makes
    """
    # As the dataset consists of only numerical column, we need not break 
    #our columns into numerical and categorical id_col is Plot_ID in this case.
    def __init__(self, dataset, file_type, train_cols, target_col, id_col, rand_seed=None):
        
        #self.data_file = data_file
        self.dataset = self._load_data(dataset, file_type) 
        
        self.target_col = target_col
        self.train_cols = train_cols
        self.id_col = id_col
        self.rand_seed = rand_seed
        self.test_ids = [] #self._create_test_ids(self.dataset, id_col, )
        self.train_set = None
        self.test_set = None
        #self.train_id = []
        
        
        self.x_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_train = pd.Series()
        self.y_test = pd.Series()
        
    def _load_data(self, data_file, file_type):
        #Loads the data from a csv file or a excel file, throws an error otherwise
        if file_type == 'csv':
            data = pd.read_csv(data_file)
    
        elif file_type == 'xlsx':
            data = pd.read_excel(data_file)
        
        else:
            return('File Type Not Supported')
        #The dataset given has two missing Plot_IDs, so fixing that by
        #repalcing them with the last twqo entries in the dataset
        data['Plot_ID'].replace(to_replace=99, value=61, inplace=True)
        data['Plot_ID'].replace(to_replace=100, value=70, inplace=True)
            
        return data        
        
    def _create_test_ids(self, col_name, num_id):
        #The function creates a list of random Plot_IDs to include in the test
        #set. This is done to prevent data leakage from train to test set due 
        #to different instances brlonging to the same Plot
        
        import random
        random.seed(self.rand_seed)
        self.test_ids = random.choices(population = np.arange(1, 99) , k = 29)
        '''    
        while len(self.test_ids) < num_id:
                test_id = random.randint(1, max(self.dataset[col_name]), random_state = 42)
                if test_id not in self.test_ids:
                        self.test_ids.append(test_id)
        '''     
         
    def create_train_test(self, num_test_ids, preprocessing = False):
        #This function uses the _create_test_ids function and the _create_xtrain_xtest
        #function to create train and test set. 
        self.test_id = self._create_test_ids( self.id_col, num_test_ids)
        self.test_set = self.dataset.loc[self.dataset[self.id_col].isin(self.test_ids)]
        self.train_set = pd.concat([self.dataset, self.test_set]).drop_duplicates(keep = False)
        self._create_xtrain_xtest(self.train_set, self.test_set, preprocessing)
    
    def _create_xtrain_xtest(self, train_set, test_set, preprocessing):
        instances = train_set[self.target_col].value_counts()
        print("Instances of train set:")
        print(instances)
        min_instances = min(self.train_set[target_col].value_counts()) 
        print("Minimum number of Instances:")
        print(min_instances)
        
        for index, value in enumerate(instances):
            print(index, value,instances.index[index] )
            if value == min_instances:
                
                temp = train_set[train_set[self.target_col] == instances.index[index]]
                self.x_train = self.x_train.append(temp)
                self.y_train = self.y_train.append(temp[self.target_col])
            
            else:
                print(train_set[train_set[self.target_col] == instances.index[index]].shape)
                temp = train_set[train_set[self.target_col] == instances.index[index]].sample(min_instances, random_state = 42 )
                print('temp')
                print(temp.shape)
                self.x_train = self.x_train.append(temp)
                #self.y_train = pd.concat([self.y_train, temp[self.target_col]], axis = 0, ignore_index = True)
                print('x_train, y_train')
                print(self.x_train)
                print(self.y_train)
        self.x_train = self.x_train.sample(frac = 1)
        self.y_train = self.x_train[self.target_col]
        self.x_train = self.x_train.iloc[:,3:]
        
        self.x_test = self.test_set.iloc[:,3:]
        self.y_test = self.test_set.iloc[:, 2]
        
        
        if preprocessing:
            self.x_train, self.x_test = self._minmax_data(self.x_train, self.x_test)
    
    def _minmax_data(self, X_train, X_test):
        # Standardization of the training and testing sets
        scaler = MinMaxScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        return X_train_sc, X_test_sc


class ModelContainer:
    
    def __init__(self, name, model):
        self.name = name
        self.model= model
        self.best_model = None
        self.classifier_gs_obj = None
        self.param_grid = None
        self.classifier_model = None
        self.predictions = None
        
    def train_model(self, data_obj , param_grid,  cv = 10,  verbose = 3, n_jobs = -1):
        
        self.param_grid = param_grid
        self.classifier_gs_obj =  GridSearchCV(self.model, self.param_grid, cv = cv,
                                          verbose=verbose, n_jobs= n_jobs)
        self.classifier_model = self.classifier_gs_obj.fit(data_obj.x_train, data_obj.y_train)
        self.best_model = self.classifier_model.best_estimator_
        
    def test_model(self, data_obj):
        
        self.predictions = self._model_predict(self.best_model, data_obj.x_test, data_obj.y_test)
        
     
    def _model_predict(self, model, X_test, y_test):
        #makig the prediction on X_test
        y_pred = model.predict(X_test)
        # Creating the confusion matrix and the classification report
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)
        print(classification_report(y_test,y_pred))
        return y_pred   
    
    def save_model(self, filename):
        pickle.dump(self.best_model, open(filename, 'wb'))
#%%
if __name__ == '__main__':
    
    dataset_file =  'Synced_bal_data_lat_paddy.csv'  
    train_cols = ['Plot_ID', 'Date', 'stage', 'B2', 'B3', 'B4', 'RE1', 'RE2','RE3','RE4', 'NIR', 'SWIR1','SWIR2', 'VH','VV'] 
    target_col = 'stage'
    id_col = 'Plot_ID'   
    rand_seed = 101
    
    # Train-Test the SVM Model    
    svm_data = Dataset(dataset_file ,'csv', train_cols, target_col , id_col, rand_seed)    
    svm_data.create_train_test(29, True)
    
    svm_model_container = ModelContainer('SVM', SVC())
    svm_grid_param = [
                    {
                     "C":  [0.1, 1, 25, 40, 50, 60, 65, 75, 100],
                     "gamma": [0.75, 0.5, 0.75, 0.9, 1, 1.2, 1.5],
                     "kernel":['linear', 'poly', 'rbf', 'sigmoid'],
                     }
                 ]
    
    svm_model_container.train_model(svm_data, svm_grid_param)
    print(svm_model_container.best_model)
    svm_model_container.test_model(svm_data)
    
    # Train-Test the Random Forest Model
        
    rf_xgb_data = Dataset(dataset_file ,'csv' , train_cols, target_col , id_col, rand_seed)    
    rf_xgb_data.create_train_test(29, False)
    
    rf_model_container = ModelContainer('Random Forest', RandomForestClassifier())

        
    rf_grid_param_band = [
                     {"max_depth":  [ 9, 10, 11, 12, 13, 14, 15],
                     "max_features": ['auto', 'sqrt', 'log2'],
                     "min_samples_leaf": [1, 2, 3],
                     "criterion": ["gini", "entropy"],
                     "n_estimators": [40, 50, 60, 100],
                     "min_samples_split":[2, 4, 5]
                      }
                    ]

    
    rf_model_container.train_model(rf_xgb_data, rf_grid_param_band)
    rf_model_container.test_model(rf_xgb_data)
    print(rf_model_container.best_model)
# Train-Test the XGboost Model
    xgb_model_container = ModelContainer('XGBoost', XGBClassifier())
    xgb_grid_param_band = [
            {
                 "eta": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                 "max_depth":  [2, 4, 6, 8, 10, 11],
                 'min_child_weight': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
                 "n_estimators": [20, 30, 40, 50, 60, 70, 80],
                 }
 
             ]
    
    xgb_model_container.train_model(rf_xgb_data, xgb_grid_param_band)
    xgb_model_container.test_model(rf_xgb_data)
    print(xgb_model_container.best_model)
    
    # Traing and testing the models on feature data
    feat_dataset_file ='Balanced_data_featuredata_paddy.xlsx' 
    feat_train_cols = ['Plot_ID', 'Date', 'stage', 'EVI', 'NDVI', 'AFRI', 'NDWI',
                  'IRECI', 'GCI', 'NPCI', 'PSRI', 'MSI', 'VVplusVH',
                  'VVbyVH', 'NRPB']     
    
    
 
    # Train-Test the SVM Model    
    svm_feat_data = Dataset(feat_dataset_file ,'xlsx', feat_train_cols, target_col , id_col, rand_seed)    
    svm_feat_data.create_train_test(29, True)
    
    svm_feat_container = ModelContainer('SVM', SVC() )
    svm_feat_grid = [
                    {
                     "C":  [0.1, 1,25,40, 50, 60, 65,75, 100],
                     "gamma": [0.01, 0.5,0.75, 0.9, 1, 1.2,1.5],
                     "kernel":['linear', 'poly', 'rbf', 'sigmoid'],
                     }
                 ]
    
    svm_feat_container.train_model(svm_feat_data, svm_feat_grid)
    print(svm_feat_container.best_model)
    svm_feat_container.test_model(svm_feat_data)
    
    svm_model_container.save_model(filename ='svm_oop_019.sav')
     # Train-Test the RF Model  
  
    rf_xgb_feat_data = Dataset(feat_dataset_file ,'xlsx', feat_train_cols, target_col , id_col, rand_seed)    
    rf_xgb_feat_data.create_train_test(29, False)
    
    rf_feat_container = ModelContainer('Random Forest', RandomForestClassifier() )
    rf_feat_grid = [
                     {"max_depth":  [ 9,10],
                      #"max_features": [2, 5, 7],
                     "max_features": ['auto', 'sqrt', 'log2'],
                     "min_samples_leaf": [1,2,3,4],
                     "criterion": ["gini", "entropy"],
                     "n_estimators": [10,40,50,60,100],
                     "min_samples_split":[2,4,5,6,7]
                     #"verbose": [1]
                     }
                    ]

    
    rf_feat_container.train_model(rf_xgb_feat_data, rf_feat_grid)
    print(rf_feat_container.best_model)
    rf_feat_container.test_model(rf_xgb_feat_data)
    
    rf_model_container.save_model(filename ='rf_oop_019.sav')
    # Train-Test the XGBoost Model  
    xgb_feat_container = ModelContainer('XGBoost-Feature', XGBClassifier())
    xgb_feat_grid = [
                    {
                      #"bootstrap":  [True],
                 "eta": [0.1,0.2],
                 #"max_depth":  [30,35,45,50],
                 "max_depth":  [4,6,7],
                 #"max_features": [3,4,5],
                 #"min_samples_leaf":  [1,2,3],
                 #"min_samples_split": [2,3,4,5],
                 'min_child_weight': [0.6,0.7, 0.8, 0.9],
                 "n_estimators": [30,40,45,50,60],
                     }
                    ]  

    
    xgb_feat_container.train_model(rf_xgb_feat_data, xgb_feat_grid)
    print(xgb_feat_container.best_model)
    rf_feat_container.test_model(rf_xgb_feat_data)
    
    
    
    
































