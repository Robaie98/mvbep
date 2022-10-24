import pandas as pd
import numpy as np
import shap 
from sklearn.model_selection import train_test_split
from .transformer import Transformer
import html


def prepare_interpretation_data(mvbep_state:dict):
    transformer = Transformer()
    initial_freq = mvbep_state['initializer']['frequency']  
    best_freq = mvbep_state['transformer']['mvbep_frequency']
    downsample = None if initial_freq == best_freq else  initial_freq+'~'+best_freq

    transformer.fit(data        = mvbep_state['initializer']['cleaned_data'], 
            timestamp_frequency = initial_freq,
            optional_features   = mvbep_state['initializer']['features'],
            occupancy_schedule  = mvbep_state['initializer']['occupancy_schedule'],
            country_code        = mvbep_state['initializer']['country_code'],
            downsample_from_to  = downsample)
    transformer.transform()

    df_train, df_test   = train_test_split(transformer.df_fin, 
            shuffle     = False,
            test_size   = mvbep_state['developer']['training_inputs']['test_size'])

    features = transformer.design_matrix_features

    return df_test, features

def return_interpretation_data(mvbep_state:dict,
                               global_sample_size:int=200, 
                               local_sample_size:int=10, 
                               df_input:pd.DataFrame = None,
                               design_matrix_features:list = None):
    # Accessing MVBEP instance state
    best_freq = mvbep_state['transformer']['mvbep_frequency']
    model_name = mvbep_state['mvbep']['best_model']
    pipe = mvbep_state['developer']['training_outputs']['frequency'][best_freq]['models_dict'][model_name]
    pipe = pipe['model']['pipe']

    # Obtaining test data
    if df_input is None:
        df_input, features = prepare_interpretation_data(mvbep_state) 
        global_sample_values = shap.sample(df_input, global_sample_size).copy()
        local_sample_values = df_input.tail(local_sample_size).copy()
    else:
        global_sample_values = df_input.head(1).copy()
        local_sample_values = df_input.copy()
        features = design_matrix_features
    
    # Fitting an explainer
    explainer = shap.KernelExplainer(pipe.predict, global_sample_values.loc[:, features].sample(1))

    # Getting scaled shap_values
    unscaled_global_shap_values = explainer.shap_values(global_sample_values.loc[:, features])
    unscaled_local_shap_values = explainer.shap_values(local_sample_values.loc[:, features])
    expected_value = explainer.expected_value

    # Creating local and global dataframes with features' values
    col_names = [col+'_effect' for col in features]
    global_sample_values.rename(columns={col:col+'_value' for col in features}, inplace=True)
    local_sample_values.rename(columns={col:col+'_value' for col in features}, inplace=True)
    global_sample_values.reset_index(drop=True, inplace=True)
    local_sample_values.reset_index(drop=True, inplace=True)

    # Creating local and global dataframes with features' effects
    global_sample_effect = pd.DataFrame(unscaled_global_shap_values, columns=col_names)
    local_sample_effect = pd.DataFrame(unscaled_local_shap_values, columns=col_names)

    # Combining effects and values
    df_plot_global = pd.concat([global_sample_values, global_sample_effect], axis=1)
    df_plot_local =  pd.concat([local_sample_values, local_sample_effect], axis=1)
    df_plot_local['e[f]'] = expected_value

    return df_plot_global, df_plot_local

