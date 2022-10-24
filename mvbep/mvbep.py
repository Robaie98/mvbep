from datetime import datetime
import tracemalloc
import pandas as pd

from .initializer import Initializer
from .transformer import Transformer
from .developer import Developer

import sys

from .writer import GenerateInitializationSummary, GenerateMVBEPSummary, GenerateQuantificationSummary
from .interpreter import return_interpretation_data

import sys
import os
import joblib


class MVBEP:
    def __init__(self, mvbep_state_path:str=None):
        self.mvbep_state = {
            'mvbep':{
                'date'             : str(datetime.now().strftime("%Y%m%d-%H%M%S")),
                'version'          : 1.0,
                'best_model'       : None,
                'development_state': 'NOT INITIATED'
            },
            
            'initializer':{
                'cleaned_data'              : None,
                'frequency'                 : None,
                'features'                  : None,
                'country_code'              : None,
                'df_validation'             : None,
                'data_sufficiency'          : None,
                'occupancy_schedule'        : None,
                'df_timestamps_highlights'  : None
            },

            'transformer':{
                'mvbep_frequency'               : None,
                'design_matrices_features'      : None
            },
            
            'developer':{
                'training_inputs':{
                    'modeling_methods'      : None,
                    'test_size'             : None,
                    'hyperparameter_tuning' : None,
                    'ranking_method'        : None
                },
                'training_outputs':{
                    'training_summary': None,
                    'frequency':{
                        '15-min':{
                            'models_dict' : None,
                            'summary':{
                                'evaluation': None, 
                                'plot_data' : None
                            }
                        },
                        'hourly':{
                            'models_dict' : None,
                            'summary':{
                                'evaluation': None, 
                                'plot_data' : None
                            }
                        },
                        'daily': {
                            'models_dict' : None,
                            'summary':{
                                'evaluation': None, 
                                'plot_data' : None
                            }
                        }
                    }
                }
            }
        }
        

        if mvbep_state_path is not None:  
            with open(mvbep_state_path, 'rb') as f:
                self.mvbep_state = joblib.load(f)
        else:
            pass


    def fit_training(self,
                    data:pd.DataFrame,
                    frequency:str,
                    country_code:str = None,
                    occupancy_schedule:dict = None,
                    mismatch_date_threshold = 0.3,
                    total_missing = None,
                    max_consec_missing = None, 
                    n_days = 360
    ): 
        # Checking the MVBEP object state
        if self.mvbep_state['mvbep']['development_state'] != 'NOT INITIATED':
            if self.mvbep_state['mvbep']['development_state'] == 'DEVELOPED':
                print("The MVBEP started the initiation process and finished developing a MVBEP model. To predict baseline values run predict_energy_consumption()")
                sys.exit()
            else:
                print("The MVBEP started the initiation but didn't start developing a MVBEP model. To start development run develop_mvbep()")
                sys.exit()

        # Creating an initializer and validating the passed data
        initializer = Initializer(mismatch_date_threshold = mismatch_date_threshold)
        initializer.fit(data = data,
                        frequency = frequency,
                        country_code = country_code,
                        occupancy_schedule = occupancy_schedule
                        )
        initializer.validate(total_missing = total_missing, 
                            max_consec_missing = max_consec_missing,
                            n_days = n_days
                            )
        
        # Updating the MVBEP object state
        if initializer.data_sufficiency:
            self.mvbep_state['mvbep']['development_state']  = 'INITIATED'
        else: 
            self.mvbep_state['mvbep']['development_state']  = 'FAILED INITIATION'
        self.mvbep_state['initializer']['cleaned_data']             = initializer.df_fin
        self.mvbep_state['initializer']['frequency']                = initializer.frequency
        self.mvbep_state['initializer']['features']                 = initializer.features
        self.mvbep_state['initializer']['country_code']             = initializer.country_code
        self.mvbep_state['initializer']['occupancy_schedule']       = initializer.occupancy_schedule
        self.mvbep_state['initializer']['df_validation']            = initializer.df_validation
        self.mvbep_state['initializer']['data_sufficiency']         = initializer.data_sufficiency
        self.mvbep_state['initializer']['df_timestamps_highlights'] = initializer.df_timestamps_highlights
         



    def generate_initialization_summary(self,
                                        file_name:str = None
    ):
        if self.mvbep_state['mvbep'] == 'NOT INITIATED':
            print('The MVBEP object has not been initiated. Initiate the model using fit_training().')
            sys.exit()
        elif self.mvbep_state['mvbep']['development_state'] == 'FAILED INITIATION':
            print("The MVBEP object failed the initiation process. Fix the data and run fit_training()")
            sys.exit()
        else: 
            GenerateInitializationSummary(file_name = file_name,
                df_input                    = self.mvbep_state['initializer']['cleaned_data'],
                frequency                   = self.mvbep_state['initializer']['frequency'],
                features                    = self.mvbep_state['initializer']['features'],
                df_timestamps_highlights    = self.mvbep_state['initializer']['df_timestamps_highlights'],
                df_validation               = self.mvbep_state['initializer']['df_validation'],
                data_sufficiency            = self.mvbep_state['initializer']['data_sufficiency'])





    def develop_mvbep(self,
                    modeling_methods:dict = None,
                    test_size:float = 0.2,
                    hyperparameter_tuning:bool = False,
                    ranking_method:str = 'min_cvrmse'#,
                    #quarter:set = (1,4) 
    ):
        # Checking initialization result 
        if self.mvbep_state['mvbep']['development_state'] == 'NOT INITIATED':
            print("The MVBEP object didn't start the initiation process. Run fit_training()")
            sys.exit()
        elif self.mvbep_state['mvbep']['development_state'] == 'FAILED INITIATION':
            print("The MVBEP object failed the initiation process. Fix the data and run fit_training()")
            sys.exit()
        elif self.mvbep_state['mvbep']['development_state'] == 'DEVELOPED':
            print("The MVBEP started the initiation process and finished developing a MVBEP model. To predict baseline values run predict_energy_consumption()")
            sys.exit()

        # Updating the MVBEP object state
        self.mvbep_state['developer']['training_inputs']['test_size']               = test_size
        self.mvbep_state['developer']['training_inputs']['hyperparameter_tuning']   = hyperparameter_tuning
        self.mvbep_state['developer']['training_inputs']['ranking_method']          = ranking_method

        # Determine possible downsampling
        downsamplings = []
        training_frequency = []
        freq_features_dict = {'15-min':None , 'hourly':None, 'daily':None, 'towt':None}
        if self.mvbep_state['initializer']['frequency'] == '15-min':
            downsamplings = [None, '15-min~hourly', '15-min~daily']
            training_frequency = ['15-min','hourly', 'daily']

        elif self.mvbep_state['initializer']['frequency'] == 'hourly':
            downsamplings = [None, 'hourly~daily']
            training_frequency = ['hourly', 'daily']
        else: 
            downsamplings = [None]
            training_frequency = ['daily']

        # MVBEP model development with different frequencies
        for downsample, freq  in zip(downsamplings, training_frequency): 
            #Transformation
            transformer = Transformer()
            transformer.fit(data                = self.mvbep_state['initializer']['cleaned_data'], 
                            timestamp_frequency = self.mvbep_state['initializer']['frequency'],
                            optional_features   = self.mvbep_state['initializer']['features'],
                            occupancy_schedule  = self.mvbep_state['initializer']['occupancy_schedule'],
                            country_code        = self.mvbep_state['initializer']['country_code'],
                            downsample_from_to  = downsample)
            transformer.transform()
            freq_features_dict[freq] = transformer.design_matrix_features
            if freq == 'hourly':
                freq_features_dict['towt'] = transformer.towt_design_matrix_features

            #Training, hyperparameter tuning, and testing
            developer = Developer(modeling_methods      = modeling_methods,
                                test_size               = test_size,
                                hyperparameter_tuning   = hyperparameter_tuning,
                                ranking_method          = ranking_method)
            self.trans_df_cehck = transformer.df_fin
            self.feat_check = transformer.design_matrix_features
            developer.fit(data                      = transformer.df_fin,
                        timestamp_frequency         = freq,
                        towt_design_matrix          = transformer.towt_design_matrix,
                        design_matrix_features      = transformer.design_matrix_features,
                        towt_design_matrix_features = transformer.towt_design_matrix_features#,
                        #quarter=quarter
                        )
            
            
            #Saving MVBEP results
            self.mvbep_state['developer']['training_inputs']['self.modeling_methods']                     = developer.modeling_methods
            self.mvbep_state['developer']['training_outputs']['frequency'][freq]['models_dict']           = developer.models_dict
            self.mvbep_state['developer']['training_outputs']['frequency'][freq]['summary']['evaluation'] = developer.show_evaluation_metrics()
            print('Done with models dict')
            self.mvbep_state['developer']['training_outputs']['frequency'][freq]['summary']['plot_data']  = developer.return_plot_data()
            print('Done with plotting')

        # Choosing the best model and best frequency
        # Summarizing outputs of each downsample iterations
        dfs_eval = []
        for freq, freq_dict in self.mvbep_state['developer']['training_outputs']['frequency'].items():
            if freq_dict['models_dict'] is not None:
               df_eval = freq_dict['summary']['evaluation'].loc[:, ['train_cvrmse', 'train_nmbe', 'test_cvrmse', 'test_nmbe']].reset_index()
               df_eval['frequency'] = freq
               dfs_eval.append(df_eval)
        training_summary = pd.concat(dfs_eval).reset_index(drop=True)

        #Saving development state
        self.mvbep_state['transformer']['design_matrices_features'] = freq_features_dict
        self.mvbep_state['developer']['training_outputs']['training_summary'] = training_summary
        condition_col = 'test_cvrmse' if ranking_method == 'min_cvrmse' else 'test_nmbe'
        self.mvbep_state['mvbep']['best_model'] = training_summary.sort_values(by=condition_col, key=abs).reset_index(drop=True)['models'][0]
        self.mvbep_state['transformer']['mvbep_frequency'] = training_summary.sort_values(by=condition_col, key=abs).reset_index(drop=True)['frequency'][0]
        self.mvbep_state['mvbep']['development_state'] = 'DEVELOPED'

    def generate_development_summary(self,
                                    file_name:str = None
    ):
        if self.mvbep_state['mvbep'] != 'DEVELOPED':
            print('The MVBEP object has not been developed. Develop the model using develop_mvbep.')
            sys.exit()
        else:
            GenerateMVBEPSummary(file_name = file_name,
                                 mvbep_state = self.mvbep_state)

    def save_mvbep_state(self, file_name:str=None):
        file_name = datetime.now().strftime("%Y%m%d-%H%M%S")+'_mvbep_state' if file_name is None else file_name
        with open(file_name, 'wb') as f:
            joblib.dump(self.mvbep_state, f, compress=5)

    def predict_energy_consumption(self,
                    data:pd.DataFrame,
                    generate_summary:bool = False,
                    file_name:str = None,
                    mismatch_date_threshold = 0.3,
                    total_missing = None,
                    max_consec_missing = None
        ):
        # Check MVBEP state
        if self.mvbep_state['mvbep']['development_state'] != 'DEVELOPED':
            print('The MVBEP object has not been developed. Develop the model using develop_mvbep.')
            sys.exit()

        # Initialization 
        mismatch_date_threshold
        initializer = Initializer(mvbep_state = self.mvbep_state,
                                  mismatch_date_threshold = mismatch_date_threshold
                                 )
        initializer.validate_pred_data(data = data,
                                       total_missing = total_missing, 
                                       max_consec_missing = max_consec_missing
                                      )

        # Transformation
        if initializer.initializer_state != 'INITIATED':
            print('The post-retrofit data failed the initialization process. Check the initialization summary.')
            GenerateQuantificationSummary(mvbep_state               = self.mvbep_state, 
                                          df_init                   = initializer.df_fin,
                                          df_savings                = None,
                                          df_timestamps_highlights  = initializer.df_timestamps_highlights,
                                          df_validation             = initializer.df_validation,
                                          data_sufficiency          = initializer.data_sufficiency,
                                          file_name                 = file_name)
            sys.exit()

        transformer = Transformer()
        data_frequency = initializer.frequency 
        best_frequency = self.mvbep_state['transformer']['mvbep_frequency']
        downsample = None if  best_frequency == data_frequency else data_frequency+'~'+best_frequency
        transformer.fit(data                = initializer.df_fin, 
                        timestamp_frequency = initializer.frequency,
                        optional_features   = initializer.features,
                        occupancy_schedule  = initializer.occupancy_schedule,
                        country_code        = initializer.country_code,
                        downsample_from_to  = downsample)
        transformer.transform()

        # Predictions 
        best_model = self.mvbep_state['mvbep']['best_model']
        pred_pipeline = self.mvbep_state['developer']['training_outputs']['frequency'][best_frequency]['models_dict'][best_model]['model']['pipe']
        if best_model == 'LR_towt':
            baseline_pred = pred_pipeline.predict(transformer.design_matrix_features)
        else: 
            baseline_pred = pred_pipeline.predict(transformer.df_fin.loc[:, transformer.design_matrix_features])
        
        # Savings
        df_savings = transformer.df_fin.copy()
        df_savings.rename(columns={'energy':'acut_post_energy'}, inplace=True)
        df_savings['base_post_energy'] = baseline_pred 

        # Interpretation
        if generate_summary:
            if not best_model.startswith('LR'):
                _ , local_shap_values = return_interpretation_data(mvbep_state=self.mvbep_state,
                                                                global_sample_size=1,
                                                                local_sample_size=df_savings.shape[0],
                                                                df_input = df_savings,
                                                                design_matrix_features= transformer.design_matrix_features)
                df_savings = local_shap_values

        # return df_savings

        # Summary 
        if generate_summary:
            GenerateQuantificationSummary(mvbep_state               = self.mvbep_state, 
                                          df_init                   = initializer.df_fin,
                                          df_savings                = df_savings,
                                          df_timestamps_highlights  = initializer.df_timestamps_highlights,
                                          df_validation             = initializer.df_validation,
                                          data_sufficiency          = initializer.data_sufficiency,
                                          file_name                 = file_name)
        else:
            return baseline_pred

        
        


    def generate_savings_summary(self):
        pass



