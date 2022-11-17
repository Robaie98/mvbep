import pandas as pd
import numpy as np 
import timeit

from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.pipeline import Pipeline


from sklearn.metrics import mean_squared_error

import statsmodels.formula.api as smf 
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

import plotly.graph_objects as go
import plotly.express as px



class Developer:

    def __init__(self,
        modeling_methods:dict = None,
        test_size:float = 0.2,
        hyperparameter_tuning:bool = False,
        ranking_method:str = 'min_cvrmse'
    ):  
        default_modeling_methods = {
            'LR' : True, # TOWT (If the frequency is hourly otherwise it is WLS)
            'RF' : True, # Random Regression Forest
            'XGB': True, # Extreme Gradient Boosting
            'SVR': True, # Support Vector Regressor
            'SLP': True, # Feed Forward Neural Network
            'KNN': True  # K-Nearest Neighbor
        }
        self.training_func = {
            'LR' : {'towt': self.develop_towt, 'wls':  self.develop_wls} ,
            'RF' : self.develop_rf,
            'XGB': self.develop_xgb,
            'SVR': self.develop_svr,
            'KNN': self.develop_knn, 
            'SLP': self.develop_slp 
        }
        
        self.modeling_methods = modeling_methods if modeling_methods is not None else default_modeling_methods 
        self.test_size = test_size
        self.hyperparameter_tuning = hyperparameter_tuning
        self.ranking_method = ranking_method
        self.timestamp_frequency = None
        self.scaled_design_matrix = None
        self.towt_design_matrix = None
        self.design_matrix_features = None
        self.towt_design_matrix_features = None

        self.rf_hyperparam = {
            'bootstrap': [True, False],
            'min_samples_leaf': [3, 5],
            'n_estimators': [100, 200, 500]
        }
        self.xgb_hyperparam = {
            'n_estimators': [100,  200, 500],
            'eta': [0.05, 0.30 ],
            'gamma': [0.0, 0.4]
        }
        self.svr_hyperparam = {
             'C': [ 0.1,  1. , 10. ],
             'kernel': ['rbf', 'poly']   
        }
        self.knn_hyperparam = {
            'n_neighbors': list(range(5,75, 5))  
        }
        self.slp_hyperparam = {
            'learning_rate_init': list(np.linspace(0.0001, 0.001, 3)),
            'hidden_layer_sizes': list(range(5, 18, 4)),
            'solver': ['adam', 'sgd'],
            'learning_rate': ['adaptive'],
            'shuffle': [False], 
            'max_iter':[300]
        }

    def fit(self, 
        data:pd.DataFrame,
        timestamp_frequency:str,
        towt_design_matrix:pd.DataFrame,
        design_matrix_features:list,
        towt_design_matrix_features:list#,
        # quarter:set
    ):
        #Defining data and features
        self.df_fin = data.copy()
        self.timestamp_frequency = timestamp_frequency
        self.towt_design_matrix = towt_design_matrix
        self.design_matrix_features = design_matrix_features
        self.towt_design_matrix_features = towt_design_matrix_features

        
        # Splitting the data into training and testing
        self.train, self.test = train_test_split(
            self.df_fin,
            test_size=self.test_size,
            shuffle=False
        )
        if self.timestamp_frequency == 'hourly':
            self.towt_train, self.towt_test = train_test_split(
                self.towt_design_matrix,
                test_size=self.test_size,
                shuffle=False
        )

        # Creating a list for timeseries cross validation indices
        self.timestamp_folds = [] 
        tss = TimeSeriesSplit(n_splits=5)
        for train_indices, val_indices in tss.split(self.train):
           self.timestamp_folds.append((train_indices, val_indices))

        #Adding Predictions to the dataframe for plotting
        train_len, test_len = self.train.shape[0], self.test.shape[0]
        self.df_fin['pred_type'] = np.concatenate((['acut_train']*train_len ,['acut_test']*test_len))

        #Training 
        self.models_dict = {}
        for key in self.modeling_methods.keys():
            if self.modeling_methods[key]:
                print('Training '+key)
                self.models_dict[key] = {}
                if self.timestamp_frequency != 'hourly':
                    if key == 'LR':
                        self.models_dict[key+'_wls'] = self.models_dict[key]
                        del self.models_dict[key]
                        self.models_dict[key+'_wls']['model'] = self.training_func[key]['wls'](
                            self.train, self.design_matrix_features
                        )
                    else: 
                        self.models_dict[key]['model'] = self.training_func[key](
                            self.train, self.design_matrix_features
                        )
                else:
                    if key == 'LR':
                        self.models_dict[key+'_towt'] = self.models_dict[key]
                        del self.models_dict[key]
                        self.models_dict[key+'_towt']['model'] = self.training_func[key]['towt'](
                            self.towt_train, self.towt_design_matrix_features
                        )
                    else:
                        self.models_dict[key]['model'] = self.training_func[key](
                            self.train, self.design_matrix_features
                        )

        #Evaluation
        for key in self.models_dict.keys():
            self.models_dict[key]['evaluation'] = {}
            model = self.models_dict[key]['model']
            if self.timestamp_frequency != 'hourly':
                estimator = model['pipe']
                y_train_acut = pd.Series.to_numpy(self.train['energy'])
                y_train_pred = estimator.predict(self.train.loc[:, self.design_matrix_features])
                y_test_acut  = pd.Series.to_numpy(self.test['energy'])
                y_test_pred  = estimator.predict(self.test.loc[:, self.design_matrix_features])
            else:
                if key != 'LR_towt':
                    estimator = model['pipe']
                    y_train_acut = pd.Series.to_numpy(self.train['energy'])
                    y_train_pred = estimator.predict(self.train.loc[:, self.design_matrix_features])
                    y_test_acut  = pd.Series.to_numpy(self.test['energy'])
                    y_test_pred  = estimator.predict(self.test.loc[:, self.design_matrix_features])
                else:
                    estimator = model['pipe']
                    y_train_acut = pd.Series.to_numpy(self.towt_train['energy'])
                    y_train_pred = pd.Series.to_numpy(estimator.predict(self.towt_train.loc[:, self.towt_design_matrix_features]))
                    y_test_acut = pd.Series.to_numpy(self.towt_test['energy'])
                    y_test_pred = pd.Series.to_numpy(estimator.predict(self.towt_test.loc[:, self.towt_design_matrix_features]))

            self.models_dict[key]['evaluation']['training'] = self.evaluate(y_pred=y_train_pred, y_acut=y_train_acut)
            self.models_dict[key]['evaluation']['testing'] = self.evaluate(y_pred=y_test_pred, y_acut=y_test_acut)
            self.df_fin[key] = np.concatenate((y_train_pred, y_test_pred))
            

            
    def develop_towt(self, train_towt:pd.DataFrame, towt_design_matrix_features:list):
        tic=timeit.default_timer()
        temp_segments = 'occupancy*' + '+ occupancy*'.join(
            feat for feat in towt_design_matrix_features if feat not in ['hours_of_week', 'occupancy'])
        towt_formula = f'energy ~ C(hour_of_week) -1 + {temp_segments}'
        towt_model = smf.wls(formula = towt_formula, data=train_towt).fit()
        toc=timeit.default_timer()
        towt_dict = {
            'pipe': towt_model,
            'hyperpar_tuning_result': None,
            'elapsed_time': toc - tic
        }
        return towt_dict

    def develop_wls(self, train:pd.DataFrame, design_matrix_features:pd.DataFrame):
        tic=timeit.default_timer()
        wls_features = '+'.join(design_matrix_features)
        wls_formula = f'energy ~ {wls_features}'
        wls_model = smf.wls(formula = wls_formula, data = train).fit()
        toc=timeit.default_timer()
        wls_dict = {
            'pipe':wls_model,
            'hyperpar_tuning_result': None,
            'elapsed_time': toc - tic
        }
        return wls_dict

    def develop_rf(self, train:pd.DataFrame, design_matrix_features:list):
        tic=timeit.default_timer()
        rf_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('rf', RandomForestRegressor(n_jobs=-1, bootstrap=True, min_samples_leaf=3, n_estimators=100))
            ])
        rf_pipe = TransformedTargetRegressor(regressor=rf_pipe, transformer=StandardScaler())
        if self.hyperparameter_tuning:
            grid_hyperparam = {'regressor__rf__'+key: self.rf_hyperparam[key] for key in self.rf_hyperparam}
            rf_grid = GridSearchCV(
                estimator = rf_pipe,
                param_grid=grid_hyperparam,
                cv = self.timestamp_folds,
                scoring = 'neg_root_mean_squared_error'
            )
            rf_grid.fit(train.loc[:, design_matrix_features], train['energy'])
            rf_pipe = rf_grid.best_estimator_
            rf_tuning_result = rf_grid.cv_results_
        else:
            rf_pipe = rf_pipe.fit(train.loc[:, design_matrix_features], train['energy'])
        toc=timeit.default_timer()
        rf_dict = {
            'pipe': rf_pipe,
            'hyperpar_tuning_result': rf_tuning_result if self.hyperparameter_tuning else None,
            'elapsed_time': toc - tic
        }
        return rf_dict

    def develop_xgb(self, train:pd.DataFrame, design_matrix_features:list):
        tic=timeit.default_timer()
        xgb_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('xgb', xgb.XGBRegressor(n_estimators=100, eta=0.05, gamma=0.4))
            ])
        xgb_pipe = TransformedTargetRegressor(regressor=xgb_pipe, transformer=StandardScaler())
        if self.hyperparameter_tuning:
            grid_hyperparam = {'regressor__xgb__'+key: self.xgb_hyperparam[key] for key in self.xgb_hyperparam}
            xgb_grid = GridSearchCV(
                estimator = xgb_pipe,
                param_grid=grid_hyperparam,
                cv = self.timestamp_folds,
                scoring = 'neg_root_mean_squared_error'
            )
            xgb_grid.fit(train.loc[:, design_matrix_features], train['energy'])
            xgb_pipe = xgb_grid.best_estimator_
            xgb_tuning_result = xgb_grid.cv_results_
        else:
            xgb_pipe = xgb_pipe.fit(train.loc[:, design_matrix_features], train['energy'])
        toc=timeit.default_timer()
        xgb_dict = {
            'pipe': xgb_pipe,
            'hyperpar_tuning_result': xgb_tuning_result if self.hyperparameter_tuning else None,
            'elapsed_time': toc - tic
        }
        return xgb_dict

    def develop_svr(self, train:pd.DataFrame, design_matrix_features:list):
        tic=timeit.default_timer()
        svr_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(C=0.1, kernel='rbf'))
            ])
        svr_pipe = TransformedTargetRegressor(regressor=svr_pipe, transformer=StandardScaler())
        if self.hyperparameter_tuning:
            grid_hyperparam = {'regressor__svr__'+key: self.svr_hyperparam[key] for key in self.svr_hyperparam}
            svr_grid = GridSearchCV(
                estimator = svr_pipe,
                param_grid=grid_hyperparam,
                cv = self.timestamp_folds,
                scoring = 'neg_root_mean_squared_error'
            )
            svr_grid.fit(train.loc[:, design_matrix_features], train['energy'])
            svr_pipe = svr_grid.best_estimator_
            svr_tuning_result = svr_grid.cv_results_
        else:
            svr_pipe = svr_pipe.fit(train.loc[:, design_matrix_features], train['energy'])
        toc=timeit.default_timer()
        svr_dict = {
            'pipe': svr_pipe,
            'hyperpar_tuning_result': svr_tuning_result if self.hyperparameter_tuning else None,
            'elapsed_time': toc - tic
        }
        return svr_dict

    def develop_slp(self, train:pd.DataFrame, design_matrix_features:list):
        tic=timeit.default_timer()
        slp_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('slp', MLPRegressor(learning_rate_init=0.00055, hidden_layer_sizes=13, solver='sgd'))
            ])
        slp_pipe = TransformedTargetRegressor(regressor=slp_pipe, transformer=StandardScaler())
        if self.hyperparameter_tuning:
            grid_hyperparam = {'regressor__slp__'+key: self.slp_hyperparam[key] for key in self.slp_hyperparam}
            slp_grid = GridSearchCV(
                estimator = slp_pipe,
                param_grid=grid_hyperparam,
                cv = self.timestamp_folds,
                scoring = 'neg_root_mean_squared_error'
            )
            slp_grid.fit(train.loc[:, design_matrix_features], train['energy'])
            slp_pipe = slp_grid.best_estimator_
            slp_tuning_result = slp_grid.cv_results_
        else:
            slp_pipe = slp_pipe.fit(train.loc[:, design_matrix_features], train['energy'])
        toc=timeit.default_timer()
        slp_dict = {
            'pipe': slp_pipe,
            'hyperpar_tuning_result': slp_tuning_result if self.hyperparameter_tuning else None,
            'elapsed_time': toc - tic
        }
        return slp_dict
        
    def develop_knn(self, train:pd.DataFrame, design_matrix_features:list):
        tic=timeit.default_timer()
        knn_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsRegressor())
            ])
        knn_pipe = TransformedTargetRegressor(regressor=knn_pipe, transformer=StandardScaler())
        grid_hyperparam = {'regressor__knn__'+key: self.knn_hyperparam[key] for key in self.knn_hyperparam}
        knn_grid = GridSearchCV(
            estimator = knn_pipe,
            param_grid= grid_hyperparam,
            cv = self.timestamp_folds,
            scoring = 'neg_root_mean_squared_error'
        )
        knn_grid.fit(train.loc[:, design_matrix_features], train['energy'])
        knn_estimator = knn_grid.best_estimator_
        knn_tuning_result = knn_grid.cv_results_
        toc=timeit.default_timer()
        knn_dict = {
        'pipe': knn_estimator,
        'hyperpar_tuning_result': knn_tuning_result,
        'elapsed_time': toc - tic
        }
        return knn_dict

    
    def evaluate(self, y_pred, y_acut):
        eval_dict = {}
        # CV(RMSE)
        eval_dict['cvrmse'] = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_acut))/np.mean(y_acut) * 100
        eval_dict['cvrmse'] = round(eval_dict['cvrmse'], 3)

        # NMBE
        eval_dict['nmbe'] = 100 * (np.sum(y_acut - y_pred) / len(y_acut)) / np.mean(y_acut)
        eval_dict['nmbe'] = round(eval_dict['nmbe'], 3)

        return eval_dict

    def show_evaluation_metrics(self):
        df_eval = pd.DataFrame.from_dict({model:self.models_dict[model]['evaluation'] for model in self.models_dict.keys()}).transpose()
        df_eval = pd.concat([
                    pd.DataFrame({'models':df_eval.index}),
                    pd.json_normalize(df_eval['training']).rename(columns={'cvrmse':'train_cvrmse', 'nmbe':'train_nmbe'}),
                    pd.json_normalize(df_eval['testing']).rename(columns={'cvrmse':'test_cvrmse', 'nmbe':'test_nmbe'})
                ], axis=1).set_index('models')
        condition_col = 'test_cvrmse' if self.ranking_method == 'min_cvrmse' else 'test_nmbe'
        ranking = [sorted(abs(df_eval[condition_col])).index(x)+1 for x in abs(df_eval[condition_col])]
        
        elapsed_time = [self.models_dict[model]['model']['elapsed_time'] for model in self.models_dict.keys()]
        
        hyperparam_tuning_state = ['Not Performed' if self.models_dict[model]['model']['hyperpar_tuning_result'] is None
                                else 'Performed' for model in self.models_dict.keys()]
        
        combinations_count = [self.models_dict[model]['model']['hyperpar_tuning_result'] for model in self.models_dict.keys()] 
        combinations_count = [1 if param_dict is None else len(param_dict['params']) for param_dict in combinations_count]
        
        df_eval['ranking'] = ranking
        df_eval['hyperparam_tuning_state'] = hyperparam_tuning_state
        df_eval['hyperparam_combin_count']  = combinations_count
        df_eval['avg_training_time'] = elapsed_time / df_eval['hyperparam_combin_count'] 
        df_eval['total_development_time']  = elapsed_time  
        return df_eval

    def return_plot_data(self):
        col_names = ['timestamp', 'energy', 'pred_type']
        col_names.extend([model for model in self.models_dict.keys()])
        df_plot = self.df_fin.loc[:, col_names]
        return df_plot
