import pandas as pd 
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os 
import re
import shutil
from .interpreter import return_interpretation_data



timestamp = {
            '15-min': {
                'type':'timedelta64[m]', 
                'timedelta': {'minutes':15},
                'freq':'15T'
            },
            'hourly':{
                'type':'timedelta64[h]', 
                'timedelta': {'hours':1},
                'freq':'h'
            },
            'daily':{
                'type':'timedelta64[D]', 
                'timedelta': {'days':1},
                'freq':'D'
            }
        }


def plot_energy_temp(df_input):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df_input['timestamp'], y=df_input['energy'], name = 'Energy'),
                    secondary_y=False)
        fig.add_trace(go.Scatter(x=df_input['timestamp'], y=df_input['dry_temp'], name = 'Dry-bulb temp'),
                    secondary_y=True)
        
        fig.update_layout(title='Energy consumption and dry-bulb temperature vs. Time', xaxis_title='Time')
        fig.update_yaxes(title_text="Energy consumption", secondary_y=False)
        fig.update_yaxes(title_text="Dry-bulb temperature", secondary_y=True)
        return fig

def plot_feature(df_input, col_name, freq_unit):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_input['timestamp'],
                                y=df_input[col_name],
                                name=col_name,
                                hovertext=df_input[col_name+'_mark']))
        
        #Highlighting irregular values 
        df_irreg = df_input[df_input[col_name+'_mark'] != 'valid']
        timedelta, delta_value = list(freq_unit.items())[0]
        cumsum = ((df_irreg['timestamp'] - df_irreg['timestamp'].shift()) != pd.Timedelta(delta_value, unit=timedelta)).cumsum()
        df_plot = df_irreg.groupby(cumsum).agg({"timestamp" : ["min", "max", 'count']})
        
        for min_time, max_time, count in zip(df_plot['timestamp']['min'], df_plot['timestamp']['max'], df_plot['timestamp']['count']):
            fig.add_vrect(x0=min_time ,
                        x1=max_time,
                        fillcolor="red", opacity=0.5,
                        layer="above", line_width=0)
            
        fig.update_layout(title=f'{col_name} vs. Time',
                        xaxis_title='Time',
                        yaxis_title=col_name,
                        hovermode="x unified")     
        return fig
    
def GenerateInitializationSummary(df_input,
                                    frequency,
                                    features,
                                    df_timestamps_highlights,
                                    df_validation,
                                    data_sufficiency,
                                    file_name=None
):

    file_name = datetime.now().strftime("%Y%m%d-%H%M%S")+'_init_sum' if file_name is None else file_name
    dr_path = os.path.dirname(os.path.realpath(__file__))+r'/templates/init_temp.html'

    #Copying initialization template
    file = ' '
    shutil.copyfile(src=dr_path, dst=file_name+'.html')
    with open(file_name+'.html', 'r', encoding='utf-8') as f:
        file = f.read() 

    #Writing metadata
    file = file.replace('attrib_time_', str(datetime.now().strftime("%Y/%m/%d - %H:%M:%S")))
    file = file.replace('attrib_start_', str(df_input['timestamp'].min()))
    file = file.replace('attrib_end_', str(df_input['timestamp'].max()))
    file = file.replace('attrib_freq_', str(frequency))
    file = file.replace('attrib_observ_count_', str(df_input.shape[0]))
    file = file.replace('attrib_feat_count_', str(len(features)))

    #Descriptive summary
    data_description = df_input.loc[:, ['timestamp']+features].describe(include='all', datetime_is_numeric=True)
    file = file.replace('attrib_descrip_summ_', data_description.to_html())

    #Data Validation
    file = file.replace('attrib_validation_timestamps_', df_timestamps_highlights.to_html())
    file = file.replace('attrib_validation_features_', df_validation.to_html())

    #Exploratory data analysis
    timedelta = timestamp[frequency]['timedelta']
    fig_energy_temp = plot_energy_temp(df_input).to_html(full_html=False)
    fig_energy_impute = plot_feature(df_input, 'energy', timedelta).to_html(full_html=False)
    fig_temp_impute = plot_feature(df_input, 'dry_temp', timedelta).to_html(full_html=False)

    file = file.replace('attrib_eda_energy_temp_', fig_energy_temp)
    file = file.replace('attrib_eda_energy_invalid_', fig_energy_impute)
    file = file.replace('attrib_eda_temp_invalid_', fig_temp_impute)

    #Final validation result
    if data_sufficiency:
        file = file.replace('attrib_validation_result', 'SUCCESS!!')
        file = file.replace('attrib_color_', 'green')
    else:
        file = file.replace('attrib_validation_result', 'Failure!!')
        file = file.replace('attrib_color_', 'red')

    #Saving the summary
    with open(file_name+'.html', 'w', encoding='utf-8') as f:
        f.write(file)

def plot_models_timeseries(df_plot, models_dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
                    x=df_plot[df_plot.pred_type=='acut_train']['timestamp'],
                    y=df_plot[df_plot.pred_type=='acut_train']['energy'], name='Actual Train'
                    ))
    fig.add_trace(go.Scatter(
                    x=df_plot[df_plot.pred_type=='acut_test']['timestamp'],
                    y=df_plot[df_plot.pred_type=='acut_test']['energy'], name='Actual Test'
                    ))
    for model in models_dict.keys():
        fig.add_trace(go.Scatter(
            x=df_plot['timestamp'], y=df_plot[model], name=model
            ))
    fig.update_layout(title='Actual vs Models Predictions in Training and Testing Data',
                xaxis_title='Time',
                yaxis_title='Energy Consumption')
    return fig

def plot_feature_importance(df_plot, transformer_features):
    features = [feat+'_effect' for feat in transformer_features]
    fig = px.box(df_plot.loc[:,features].rename(columns={col:col.replace('_effect','') for col in features}) ,
                    orientation='h')
    fig.update_layout(
        title='Feature Importance Summary by SHAP Values',
        xaxis_title = 'SHAP Value',
        yaxis_title = 'Feature',
        hovermode=False)
    return fig

def plot_temp_effect_vs_energy(df_plot):
    fig = px.scatter(df_plot, x="dry_temp_value", y='energy', color='dry_temp_effect')
    fig.update_layout(
        title='Outdoor Dry-bulb Temperature Vs Energy with SHAP values',
        xaxis_title = 'Outdoor Dry-bulb Temperature',
        yaxis_title = 'Energy Consumption',
        hovermode=False)
    return fig

def plot_features_timeseries_effect(df_plot, pre_retrofit=True):
    fig = go.Figure()
    feats = [feat for feat in df_plot.columns if feat.endswith('_effect')] + ['e[f]']
    
    if pre_retrofit:
        fig.add_trace(go.Scatter(
                        x=df_plot['timestamp'],
                        y=df_plot.loc[:,'energy'], name='Actual Energy Consumption'
                      ))
    
    fig.add_trace(go.Scatter(
                    x=df_plot['timestamp'],
                    y=df_plot.loc[:,feats].sum(axis=1), name='Predicted Energy Consumption'
                    ))
    
    for col in df_plot.keys():
        if col.endswith('_effect'):
            fig.add_trace(go.Scatter(
                x=df_plot['timestamp'], y=df_plot[col], name=col
                ))
    fig.update_layout(title='Features Additive Effect on Predicted Energy Consumption',
                xaxis_title='Time',
                yaxis_title='Predicted Energy Consumption')
    return fig

def plot_baseline_vs_post(df_plot):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
                    x=df_plot['timestamp'],
                    y=df_plot['acut_post_energy'],
                    name = 'Actual Post-Retrofit'
                    ))
    fig.add_trace(go.Scatter(
                    x=df_plot['timestamp'],
                    y=df_plot['base_post_energy'],
                    name = 'Baseline Post-Retrofit'
                    ))
    fig.update_layout(title='Actual and Baseline Post-retrofit Energy Consumption',
                xaxis_title='Time',
                yaxis_title='Energy Consumption')
    return fig

def savings_timebars(df_input):
    df_plot = df_input.copy()
    df_plot['Savings'] = df_plot['base_post_energy'] - df_plot['acut_post_energy']
    df_plot.rename(columns={'base_post_energy':'Baseline', 'acut_post_energy':'Actual'})
    df_plot['Savings State'] = df_plot['Savings'].apply(lambda x: 'Negative' if x < 0 else 'Positive')
    
    fig = px.bar(df_plot, x='timestamp',
                 y='Savings',
                 color='Savings State',
                 color_discrete_map= {'Negative':'red', 'Positive':'green'}
                )
    
    fig.update_layout(title='Energy Savings during Post-Retrofit Period',
                xaxis_title='Time',
                yaxis_title='Energy Savings')
    
    return fig


def GenerateMVBEPSummary(mvbep_state:dict,
                        file_name:str = None
):
    file_name = datetime.now().strftime("%Y%m%d-%H%M%S")+'_dev_sum' if file_name is None else file_name
    dr_path = os.path.dirname(os.path.realpath(__file__))+r'/templates/dev_temp.html'

    # Copying initialization template
    file = ' '
    shutil.copyfile(src=dr_path, dst=file_name+'.html')
    with open(file_name+'.html', 'r', encoding='utf-8') as f:
        file = f.read() 

    # Initializer
    ## Description
    init_dict = mvbep_state['initializer']
    file = file.replace('<attrib_time_>', str(datetime.now().strftime("%Y/%m/%d - %H:%M:%S")))
    file = file.replace('<attrib_start_>', str(init_dict['cleaned_data']['timestamp'].min()))
    file = file.replace('<attrib_end_>', str(init_dict['cleaned_data']['timestamp'].max()))
    file = file.replace('<attrib_freq_>', str(init_dict['frequency']))
    file = file.replace('<attrib_observ_count_>', str(init_dict['cleaned_data'].shape[0]))
    file = file.replace('<attrib_feat_count_>', str(len(init_dict['features'])))
    ## Descriptive summary
    data_description = init_dict['cleaned_data'].loc[:, ['timestamp']+init_dict['features']].describe(include='all', datetime_is_numeric=True)
    file = file.replace('<attrib_descrip_summ_>', data_description.to_html())
    ## Data Validation
    file = file.replace('<attrib_validation_timestamps_>', init_dict['df_timestamps_highlights'].to_html())
    file = file.replace('<attrib_validation_features_>', init_dict['df_validation'].to_html())
    ## Exploratory data analysis
    fig_energy_temp = plot_energy_temp(init_dict['cleaned_data']).to_html(full_html=False)
    timedelta = timestamp[init_dict['frequency']]['timedelta']
    fig_energy_impute = plot_feature(init_dict['cleaned_data'], 'energy', timedelta).to_html(full_html=False)
    fig_temp_impute = plot_feature(init_dict['cleaned_data'], 'dry_temp', timedelta).to_html(full_html=False)
    file = file.replace('<attrib_eda_energy_temp_>', fig_energy_temp)
    file = file.replace('<attrib_eda_energy_invalid_>', fig_energy_impute)
    file = file.replace('<attrib_eda_temp_invalid_>', fig_temp_impute)
    
    # Transformer
    ## Occupancy schedule
    # if init_dict['occupancy_schedule'] is not None:
    #     file = file.replace('<attrib_occup_schedule_>', init_dict['occupancy_schedule'])
    # else:
    #     file = file.replace('<h3>2.1 Occupancy Schedule </h3>', '') 
    # ## Holidays
    # if init_dict['country_code'] is not None:
    #     file = file.replace('<attrib_holidays_>', init_dict['country_code'])
    # else:
    #     file = file.replace('<h3>2.2 Country Holiday </h3>', '')
    #     file = re.sub('[^.>]*[?:Python]\.', '', file)

    
    # Model Development
    train_inputs = mvbep_state['developer']['training_inputs']  
    train_outputs = mvbep_state['developer']['training_outputs']
    ## Description
    if train_inputs['hyperparameter_tuning']:
        file = file.replace('<attrib_dev_hyperparam_state_>', 'performed')
    else: 
        file = file.replace('<attrib_dev_hyperparam_state_>', 'was not performed')
    file = file.replace('<attrib_dev_test_size_>', str(train_inputs['test_size']))
    ## Evaluation table
    file = file.replace('<attrib_model_eval_table_>', train_outputs['training_summary'].to_html())
    ## Evaluation plots
    # eval_plots = ''
    # plots = []
    for freq in train_outputs['frequency'].keys():
        if train_outputs['frequency'][freq]['models_dict'] is not None:
            fig = plot_models_timeseries(df_plot = train_outputs['frequency'][freq]['summary']['plot_data'],
             models_dict = train_outputs['frequency'][freq]['models_dict'])
            # plots.append(fig)
            eval_plots += f'<h4>{freq} Plot</h4>'
            eval_plots += fig.to_html(full_html=False)
    file = file.replace('<attrib_model_eval_plot_>', eval_plots)
    ## Best model
    best_model_metric = train_inputs['ranking_method'].split('_')[1]
    best_model_name = mvbep_state['mvbep']['best_model']
    best_model_freq = mvbep_state['transformer']['mvbep_frequency']
    best_model_design_matrix_features = mvbep_state['transformer']['design_matrices_features'][best_model_freq]
    file = file.replace('<attrib_best_model_metric_>', best_model_metric)
    file = file.replace('<attrib_best_model_name_>', best_model_name)
    file = file.replace('<attrib_best_model_freq_>', best_model_freq)
    best_model_dict = train_outputs['frequency'][mvbep_state['transformer']['mvbep_frequency']]
    best_model_dict = best_model_dict['models_dict'][mvbep_state['mvbep']['best_model']]

    ### Model's describtion
    if best_model_name in ['RF', 'XGB','SVR', 'SLP', 'KNN']:
        model_describe = best_model_dict['model']['pipe'].regressor_[1]
        file = file.replace('<attrib_best_model_describe_>', f'<pre> <code>{str(model_describe)}</code></pre>')
    else:
        model = best_model_dict['model']['pipe']
        model_tables = model.summary().tables
        model_tables = [table.as_html() for table in model_tables]
        model_tables = ' <br> '.join(model_tables)
        file = file.replace('<attrib_best_model_describe_>', model_tables)


    ## Hyperparameter tuning process result
    if best_model_dict['model']['hyperpar_tuning_result'] is None:
        file = file.replace('<h4> 3.2.1 Hyperparameter Tuning Process </h4>', '')
    else:
        hyperparam_result = pd.DataFrame.from_dict(best_model_dict['model']['hyperpar_tuning_result']).to_html()
        file = file.replace('attrib_best_model_hyperparam_process_', hyperparam_result)

    ## Model Interpretation
    if best_model_name in ['LR_towt', 'LR_wls']:
        file = file.replace('<attrib_best_model_fig_shap_summary>', 'The model is not complex. Hence, no interpretation is provided.')
    else:
        ### Obtaining global and local interpretation data
        df_plot_global, df_plot_local = return_interpretation_data(mvbep_state = mvbep_state)

        ### Global interpretation
        #### SHAP values summary
        fig_shap_summary = plot_feature_importance(df_plot=df_plot_global,
                                                transformer_features=best_model_design_matrix_features)
        file = file.replace('<attrib_best_model_fig_shap_summary>', fig_shap_summary.to_html(full_html=False))
        ### Dry-temp effect vs Energy consumption
        if best_model_freq != 'daily':
            fig_temp_effect_vs_energy = plot_temp_effect_vs_energy(df_plot=df_plot_global)
            file = file.replace('<attrib_best_model_fig_temp_effect_vs_energy>', fig_temp_effect_vs_energy.to_html(full_html=False))

        ### Local interpretation
        #### Features' effects and predicted energy consumption
        fig_features_timeseries_effect = plot_features_timeseries_effect(df_plot=df_plot_local)
        file = file.replace('<attrib_best_model_fig_features_timeseries_effect>', fig_features_timeseries_effect.to_html(full_html=False))

    #Saving the summary
    with open(file_name+'.html', 'w', encoding='utf-8') as f:
        f.write(file)



def GenerateQuantificationSummary(mvbep_state,
                                  df_init,
                                  df_savings,
                                  df_timestamps_highlights,
                                  df_validation,
                                  data_sufficiency,
                                  file_name=None
):
    file_name = datetime.now().strftime("%Y%m%d-%H%M%S")+'_init_sum' if file_name is None else file_name
    dr_path = os.path.dirname(os.path.realpath(__file__))+r'/templates/quant_temp.html'

    #Copying Quantification template
    file = ' '
    shutil.copyfile(src=dr_path, dst=file_name+'.html')
    with open(file_name+'.html', 'r', encoding='utf-8') as f:
        file = f.read() 

    # Obtain variables from MVBEP state
    frequency       = mvbep_state['initializer']['frequency']
    features        = mvbep_state['initializer']['features']
    best_model_name = mvbep_state['mvbep']['best_model']

    #Writing metadata
    file = file.replace('attrib_time_', str(datetime.now().strftime("%Y/%m/%d - %H:%M:%S")))
    file = file.replace('attrib_start_', str(df_init['timestamp'].min()))
    file = file.replace('attrib_end_', str(df_init['timestamp'].max()))
    file = file.replace('attrib_freq_', str(frequency))
    file = file.replace('attrib_observ_count_', str(df_init.shape[0]))
    file = file.replace('attrib_feat_count_', str(len(features)))

    #Descriptive summary
    data_description = df_init.loc[:, ['timestamp']+features].describe(include='all', datetime_is_numeric=True)
    file = file.replace('attrib_descrip_summ_', data_description.to_html())

    #Data Validation
    file = file.replace('attrib_validation_timestamps_', df_timestamps_highlights.to_html())
    file = file.replace('attrib_validation_features_', df_validation.to_html())

    #Exploratory data analysis
    timedelta = timestamp[frequency]['timedelta']
    fig_energy_temp = plot_energy_temp(df_init).to_html(full_html=False)
    fig_energy_impute = plot_feature(df_init, 'energy', timedelta).to_html(full_html=False)
    fig_temp_impute = plot_feature(df_init, 'dry_temp', timedelta).to_html(full_html=False)

    file = file.replace('attrib_eda_energy_temp_', fig_energy_temp)
    file = file.replace('attrib_eda_energy_invalid_', fig_energy_impute)
    file = file.replace('attrib_eda_temp_invalid_', fig_temp_impute)

    # Remaining variables 
    figs = []
    for feat in features:
        if feat not in ['energy', 'dry_temp']:
            figs.append(plot_feature(df_init, feat, timedelta).to_html(full_html=False))
    
    file = file .replace('attrib_eda_multiple_invalid_', '<br><br>'.join(figs))


    #Final validation result
    if data_sufficiency:
        file = file.replace('attrib_validation_result', 'SUCCESS!!')
        file = file.replace('attrib_color_', 'green')
        file = file.replace('<val_break>', ' ')
    else:
        file = file.replace('attrib_validation_result', 'Failure!!')
        file = file.replace('attrib_color_', 'red')
        file = file.split('<val_break>')[0]
        with open(file_name+'.html', 'w', encoding='utf-8') as f:
            f.write(file)


    # If the validation is successfull 
    if data_sufficiency:
        # Predicted baseline vs actual post-retrofit
        fig_baseline_post = plot_baseline_vs_post(df_savings)
        file = file.replace('<attrib_model_baseline_vs_post_>', fig_baseline_post.to_html(full_html=False))  


        # Predictions interpretation
        if best_model_name in ['LR_towt', 'LR_wls']:
            file = file.replace('<attrib_best_model_fig_features_timeseries_effect>', 'The model is not complex. Hence, no interpretation is provided.')
        else:
            #### Features' effects and predicted energy consumption
            fig_features_timeseries_effect = plot_features_timeseries_effect(df_plot=df_savings, pre_retrofit=False)
            file = file.replace('<attrib_best_model_fig_features_timeseries_effect>', fig_features_timeseries_effect.to_html(full_html=False))

        # Savings time bars 
        fig_savings_bar = savings_timebars(df_savings)
        file = file.replace('<attrib_model_savings_bars_>', fig_savings_bar.to_html(full_html=False))

        # Savings total 
        savings_total = """
                The amount of total savings between {} and {} is {}
        """.format(str(df_init['timestamp'].min()),
                    str(df_init['timestamp'].max()),
                    str((df_savings['base_post_energy'] - df_savings['acut_post_energy']).sum())
                  )
        file = file.replace('<attrib_model_savings_total_>', savings_total)
        with open(file_name+'.html', 'w', encoding='utf-8') as f:
            f.write(file)


