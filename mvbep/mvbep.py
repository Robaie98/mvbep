"""
Measurement and Verification Building Energy Prediction (MVBEP) is a `class` that encompasses different
modules for reading and validating input data to transforming such data and using them to develop regression
models for savings estimations in the post-retrofit period. 

The `class` is fitted by using `fit_training()` which takes in the required input data. Followingly, an
initialization summary is produced to check the data sufficiency requirements or the need for any actions
to fix the input data. If the data met the requirements to build a model, the function `develop_mvbep()`
is used to transform the data, train, and evaluate regression models. `GenerateDevelopmentSummary()` function 
can be used to see the summary of the development process. Finally, savings are estimated passing using post-retrofit
data to `predict_energy_consumption()` function. The current state of the documentation covers only `MVBEP` class.
Future additions to the project includes writing the documentation for the remaining modules (i.e. `Initializer`,
`Transformer`, `Developer`, 'Interpreter', and 'Writer').

Please check the provided Notebooks for the package demonstration.

"""


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
    """ MVBEP class to perform all steps of building an energy consumption baseline. 

        The class incorporates the 4 required modules for building a baseline starting 
        from initialization to savings quantification.

        Parameters
        ----------
        mvbep_state_path : str:  (Default value = None)  
            The file path for a saved MVBEP state in case the baseline creation process 
            stopped before the final step and saved by `save_mvbep_state()`.

        Attributes
        ----------
        mvbep_state : dict:  
            A python object that saves all the required information for either continuing
            the process of MVBEP or quantifying savings when the MVBEP object is developed. 
            [Check MVBEP state structure](??) 


        Example
        ----------
        In case a object of MVBEP was saved by using `save_mvbep_state` before, it can be loaded like
        ```
        mvbep_boulder = MVBEP(mvbep_state_path = 'mvbep_states/office-boulder_mvbep-state')
        ```

        In case there was no object saved before, an instance of MVBEP is created by
        ```
        mvbep_boulder = MVBEP()
        ```

    """
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
                    mismatch_date_threshold:float = 0.3,
                    total_missing = None,
                    max_consec_missing = None, 
                    n_days = 360
    ): 
        """ Fits a MVBEP object with raw data. 

        This is the first method in developing an energy consumption baseline. The
        method takes required historical data to prepare them for next processes. 

        Parameters
        ----------
        data : pd.DataFrame:   
            A dataframe that includes the required data which includes at least 
            
            - Timestamps in 15-min or hourly intervals
            - Energy consumption
            - Outdoor dry-bulb temperature
        frequency : str: {'15-min', 'hourly'}
            The timestamps intervals frequency.
        country_code : str: (default `None`)
            A two-letter `str` indicating the country code in which the building resides.
            The supported codes are listed in holiday package [documentation](https://pypi.org/project/holidays/)
        occupancy_schedule : dict: (default `None`)
            A `dict` indicating the general occupancy density in the building. [Check
            the parameter structure ](??)
        mismatch_date_threshold : float: (default = 0.3)
            Sets the threshold for values in `timestamp` column that cannot be converted from `str`
            to `datetime` object.  
        total_missing : int: (default `None`, The value is set based on frequency)
            Sets a threshold for the total number of a feature's missing observations to meet 
            data sufficiency requirements.
        max_consec_missing : int: (default `None`, The value is set based on frequency)
            Sets a threshold for consecutive missing observations in a single feature before
            the feature is dropped.
        n_days : int: (default 365)
            Sets a threshold for the least number of days in `data`. 

        Example
        ----------
        Example of a building located in Boulder, CO, USA with hourly timestamps. The instance of MVBEP
        was created with a nmae of `mvbep_boulder`.
        ```
        mvbep_boulder.fit_training(
            data = df_boulder_office, 
            frequency = 'hourly',
            country_code = 'US'
        )

        ```


        """
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
        """ Generates summary of the initialization performed after `fit_training()`.

        The initialization summary is generated as an HTML file with highlights of the 
        initialization process including plots, descriptive data, and data sufficiency result.


        Parameters
        ----------
        file_name : str: (default None)
            Sets the name of the HTML initialization summary. In case no name was provided, 
            the resulting name will be `initiation_time` + `init_sum_`.

        Example
        ----------
        Writing the initialization summary of `mvbep_boulder` after running `fit_training()`.
        ```
        mvbep_boulder.generate_initialization_summary(file_name = 'mvbep_summaries/office-boulder_init-summary')
        ```

        """ 
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
                    ranking_method:str = 'min_cvrmse'
    ):
        """ Transforms the cleaned data and develops regression models.

        Takes the cleaned data after `fit_training()` and iterates over the possible transformations
        while using each transformation to generate regression models using the chosen modeling approaches
        in `modeling_methods`. With each transformation, outputs such as evaluation metrics and models are 
        saved in the MVBEP object's state (i.e. attribute `mvbep_state`).

        Parameters
        ----------
        modeling_methods : dict:  (default None)
            The chosen modeling approaches to develop the baseline. In case None was passed, the argument is 
            passed by:
        ```python
        default_modeling_methods = {
            'LR' : True, # TOWT (If the frequency is hourly otherwise it is WLS)
            'RF' : True, # Random Regression Forest
            'XGB': True, # Extreme Gradient Boosting
            'SVR': True, # Support Vector Regressor
            'SLP': True, # Feed Forward Neural Network
            'KNN': True  # K-Nearest Neighbor
        }
        ```
        test_size : float: (default 0.2)
            Sets the testing set size out of the input data.
        hyperparameter_tuning : bool: (defalut False)
            
            - If True: the hyperparameter tuning process is performed for any model with hyperparameters to
            be tuned. 
            - If False: No hyperparameter tuning process is performed (except for KNN). 
        ranking_method : str: {'min_cvrmse', 'min_nmbe'} (default 'min_cvrmse')
            Sets the ranking method to choose the best model based on the testing set evaluation.

            - If 'min_cvrmse': The best model is selected based on Coefficient of Variation of Root Mean
            Squared Error (CV(RMSE))
            - If 'min_nmbe': The best model is selected based on Normalized Mean Bias Error (NMBE).


        Example
        ----------
        Developing `mvbep_boulder` after running `fit_training()` 
        ```
        mvbep_boulder.develop_mvbep()
        ```


        """ 
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
            self.mvbep_state['developer']['training_outputs']['frequency'][freq]['summary']['plot_data']  = developer.return_plot_data()

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
        """ Generates development summary after using `develop_mvbep()`.

        Outputs an HTML file that summarizes the development process after running `develop_mvbep()`  

        Parameters
        ----------
        file_name : str: (default None)
            Sets the name of the HTML development summary. In case no name was provided, 
            the resulting name will be `initiation_time` + `dev_sum_`.

        Example
        ----------
        Writing the initialization summary of `mvbep_boulder` after running `develop_mvbep()`.
        ```
        mvbep_boulder.generate_development_summary(file_name = 'mvbep_summaries/office-boulder_dev-summary')
        ```


        """ 
        if self.mvbep_state['mvbep']['development_state'] != 'DEVELOPED':
            print('The MVBEP object has not been developed. Develop the model using develop_mvbep.')
            sys.exit()
        else:
            GenerateMVBEPSummary(file_name = file_name,
                                 mvbep_state = self.mvbep_state)

    def save_mvbep_state(self, file_name:str=None):
        """ Saves the current progress of the MVBEP object by storing `mvbep_state`.

        Parameters
        ----------
        file_name : str: (default None)
            Sets the name of the `Joblib` state file. In case no name was provided, 
            the resulting name will be `initiation_time` + `mvbep_state`.
            
        Example
        ----------
        Saving the state of either an initiated MVBEP by `fit_training()` or a developed one by `develop_mvbep()`.
        ```
        mvbep.save_state('mvbep_states/office-boulder_mvbep-state')
        ```


        """ 
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
        """ Generates savings quantification summary after using `develop_mvbep()`.

        Outputs an HTML file that summarizes the quantification process after running `develop_mvbep().`
        The quantification process requires post-retrofit data that matches the same frequency and features
        of the data used in initialization when running `fit_training()`. Features that was dropped in the
        initialization process are not required in the post-retrofit data. To see which features passed the 
        initialization process, check the output of `generate_initialization_summary()`.

        Parameters
        ----------
        data :pd.DataFrame 
            The post-retrofit data.
        generate_summary :bool: (default False)
            Either generates a summary in an HTML file or return a `list` of baseline energy consumption.
            In case the passed `data` does not meet the requirements, an initialization summary is generated
            regardless of the passed argument in `generate_summary`.

            - If True: A quantification summary is provided. The function does not return any object.
            - If False: A list of baseline energy consumption for the provided post-retrofit period is 
            returend.  
        file_name : str: (default None)
            Sets the name of the HTML quantification summary. In case no name was provided, 
            the resulting name will be `initiation_time` + `quant_sum_`.
        mismatch_date_threshold : float: (default = 0.3)
            Sets the threshold for values in `timestamp` column that cannot be converted from `str`
            to `datetime` object.  
        total_missing : int: (default `None`, The value is set based on frequency)
            Sets a threshold for the total number of a feature's missing observations to meet 
            data sufficiency requirements.
        max_consec_missing : int: (default `None`, The value is set based on frequency)
            Sets a threshold for consecutive missing observations in a single feature before
            the feature is dropped. 

        Example
        ----------
        Writing the quantification summary of `mvbep_boulder`.
        ```
        mvbep_boulder.predict_energy_consumption(data = df_boulder_post_retrofit,
                                                generate_summary = True,
                                                file_name='mvbep_summaries/office-boulder_dev-summary')
        ```


        """ 
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




