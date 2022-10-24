import pandas as pd
import numpy as np 
from datetime import datetime
import holidays
from schema import Schema, Or
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os 
import sys
import shutil

class Initializer:

    def __init__(
        self,
        mismatch_date_threshold = 0.3,
        mvbep_state:dict = None
        ):
        
        self.data_min_feats = ['timestamp','energy', 'dry_temp'] 
        self.raw_df = None
        self.mismatch_date_threshold = mismatch_date_threshold 
        self.features = None
        self.occupancy_schedule_schema = Schema({
            'annually': Or(list, None), 
            'weekly': Or(   {
                                0: Or(dict, None),
                                1: Or(dict, None),
                                2: Or(dict, None), 
                                3: Or(dict, None),
                                4: Or(dict, None),
                                5: Or(dict, None),
                                6: Or(dict, None)  
                            },
                            0
                        )
        })
        self.timestamp = {
            '15-min': {
                'type':'timedelta64[m]', 
                'timedelta': {'minutes':15},
                'freq':'15min'
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
        self.validation_criteria = {
            '15-min':{
                'total_missing': 3500, #10% of total possible values count in a year 
                'max_consec_missing': 96, #Total possible values count in a day
            },
            'hourly':{
                'total_missing': 876, #10% of total possible values count in a year 
                'max_consec_missing': 24 #Total possible values count in a day
            },
            'daily':{
                'total_missing': 37, #10% of total possible values count in a year 
                'max_consec_missing': 7 #Total possible values count in a week
            }
        }
        self.highlights_features = {}

        if mvbep_state is not None:
            self.frequency          = mvbep_state['initializer']['frequency']
            self.country_code       = mvbep_state['initializer']['country_code']
            self.occupancy_schedule = mvbep_state['initializer']['occupancy_schedule']
            self.data_min_feats     = mvbep_state['initializer']['features']
            self.last_date          = mvbep_state['initializer']['cleaned_data']['timestamp'].max()
            self.initializer_state = 'INITIATED'
        else:
            self.initializer_state = 'NOT INITIATED'



    def convert_2_numeric(self, series, conv_type=float):
        conv_series = series.astype(conv_type, errors='ignore')
        conv_series = conv_series.apply(lambda value: np.nan if type(value) != conv_type else value)
        return conv_series

    def fit(
        self,    
        data: pd.DataFrame,
        frequency: str,
        country_code: str = None,
        occupancy_schedule: dict = None
    ):
        # Check for minimum features
        self.frequency = frequency
        raw_feats = data.columns.tolist()
        if all(feat in raw_feats for feat in self.data_min_feats):
            self.df_raw = data.copy()
        else: 
            missing_feats = [feat for feat in self.data_min_feats if feat not in raw_feats]
            raise Exception('Your passed data is missing {}'.format(missing_feats))

        # Checking passed date format
        self.df_fin = self.df_raw.copy()
        self.df_fin['timestamp'] = pd.to_datetime(self.df_fin['timestamp'],
                                                     errors='coerce')
        datetime_mismatch_percent = self.df_fin['timestamp'].isna().sum()
        datetime_mismatch_percent = datetime_mismatch_percent / len(self.df_fin['timestamp']) 
        if self.mismatch_date_threshold > datetime_mismatch_percent:
            pass
        else:
            raise Exception('The percent of date format mismatch ({}) exceeds the limit of {} '.format(datetime_mismatch_percent, self.mismatch_date_threshold)) 

        # Changing data types
        self.features = [feat for feat in self.df_fin.columns if feat != 'timestamp']
        for feat in self.features:
            self.df_fin[feat] = self.convert_2_numeric(self.df_fin[feat])

        # Checking country code
        if country_code is not None:
            try:
                holidays.country_holidays(country_code, years=2015)
                self.country_code = country_code
            except:
                print('Country code or name does not exist')
                self.country_code = None
        else: 
            self.country_code = None
                
        # Checking occupancy schedule 
        if occupancy_schedule is not None:
            occupancy_schedule_pass = self.occupancy_schedule_schema.is_valid(occupancy_schedule)
            if occupancy_schedule_pass:
                self.occupancy_schedule = occupancy_schedule
            else:
                self.occupancy_schedule = None
                print("The passed occupancy_schedule's schema does not match the format")
        else:
            self.occupancy_schedule = None

    def fill_gaps(self, series, frequency):
        first = series.min(); last = series.max()
        new_timestamps = pd.date_range(start=first, end=last, freq=self.timestamp[frequency]['freq'])
        new_timestamps = pd.Series(new_timestamps).to_frame('timestamp')
        new_timestamps = new_timestamps.merge(series.to_frame(), left_on='timestamp',  right_on='timestamp', how='outer', indicator=True)
        new_timestamps.replace({'both':'valid', 'left_only':'jump'}, inplace=True)
        new_timestamps.rename(columns={'_merge':'timestamp_mark'}, inplace=True)
        return new_timestamps

    def consec_count(self, df_input, column_name):
        df_out = df_input.copy()
        df_out['missing_mark'] = df_out[column_name].isna()
        df_out['mask'] = df_out['missing_mark']==True
        g = df_out['mask'].ne(df_out['mask'].shift()).cumsum()
        df_out['count'] = df_out.groupby(g)['mask'].transform('size') * np.where(df_out['mask'], 1, -1)
        consec_miss_max = max(df_out.loc[df_out['count'] > 0, 'count'].max(), 0)
        consec_miss_max = consec_miss_max if not np.isnan(consec_miss_max) else 0
        return consec_miss_max

    def generate_feature_summary(self, feature_series:pd.Series, mark_value=False):
        name = feature_series.name
        summary_dict = {name:{}}
        
        #Missing values
        mask, mask_count = feature_series.isna(), sum(feature_series.isna())
        summary_dict[name]['miss_values'] = mask_count
        if mark_value:
            self.df_fin[name+'_mark'] = 'valid'
            self.df_fin.loc[mask, name+'_mark'] = ['miss_imputated']*mask_count
        
        #Invalid values (Negative or zero)
        if name == 'energy':
            mask, mask_count = feature_series <=0 , sum(feature_series <=0)
            summary_dict[name]['invalid_values'] = mask_count
            self.df_fin.loc[mask, name] = [np.nan]*mask_count
            if mark_value:
                self.df_fin.loc[mask, name+'_mark'] = ['invalid_imputated']*mask_count
        else:
            summary_dict[name]['invalid_values'] = 0

        #Total missing 
        mask, mask_count = feature_series.isna(), sum(feature_series.isna())
        summary_dict[name]['total_missing_values'] = mask_count

        #Consecutive jumps
        summary_dict[name]['max_consec_missing_values'] = self.consec_count(self.df_fin, name)

        #Imputation
        self.df_fin[name] = self.df_fin[name].interpolate()
        
        return summary_dict

    def generate_timestamp_summary(self, df_input):
        df_out = df_input.copy()
        highlight_timestamp = {} 

        ## Format mismatch
        tot_obs_raw_data = df_out.shape[0] 
        df_out.dropna(subset=['timestamp'], inplace=True)
        highlight_timestamp['timestamp_format_mismatch'] = tot_obs_raw_data - df_out.shape[0]
        
        ## Jumps
        fixed_timestamps = self.fill_gaps(df_out['timestamp'], frequency=self.frequency)
        df_out = df_out.merge(fixed_timestamps, left_on='timestamp', right_on ='timestamp', how='outer')
        highlight_timestamp['timestamp_jumps'] = df_out.loc[df_out['timestamp_mark']=='jump','timestamp_mark'].count()
        
        ## Consecuitive Missing
        highlight_timestamp['max_timestamp_consec_jumps'] = self.consec_count(df_out, 'timestamp')
        
        ## Period of fitted data
        highlight_timestamp['timestamp_range'] = (df_out['timestamp'].max() - df_out['timestamp'].min()).days 
        
        return df_out, highlight_timestamp


    def validate(self, 
        total_missing = None, 
        max_consec_missing = None,
        n_days = 360
    ):
        # Checking Initiation State
        if self.initializer_state == 'INITIATED':
            print('The intitializer object has been initiated already.')
            sys.exit()
        elif self.initializer_state == 'FAILED INITIATION':
            print('The intitializer object failed the initiation. Fix the data and run fit()')
            sys.exit()

        # Validation criteria
        self.validation_criteria[self.frequency]['total_missing'] = self.validation_criteria[self.frequency]['total_missing'] if total_missing is None else total_missing  
        self.validation_criteria[self.frequency]['max_consec_missing'] = self.validation_criteria[self.frequency]['max_consec_missing'] if max_consec_missing is None else max_consec_missing  

        # Timestamps validation
        self.df_fin, highlights_timestamps = self.generate_timestamp_summary(self.df_fin)

        # Other features' validation
        highlights_features = []
        for feat in self.features:
            feat_series = self.df_fin[feat]
            highlights_features.append(self.generate_feature_summary(feat_series, mark_value=True))

        # Creating highlights dataframes
        # Timestamps   
        self.df_timestamps_highlights = pd.DataFrame(
            index = highlights_timestamps.keys(), columns=['result'], data=highlights_timestamps.values()
        )

        # Other features
        indices = [list(feat.keys())[0] for feat in highlights_features]
        cols = list(highlights_features[0][indices[0]].keys())
        rows = [highlights_features[i][indices[i]] for i in range(len(indices))]
        rows = [list(row_dict.values()) for row_dict in rows]
        df_features_highlights = pd.DataFrame(index=indices, columns=cols, data=rows).transpose()

        #Checking validation requirements 
        self.df_validation = df_features_highlights.loc[['total_missing_values', 'max_consec_missing_values'], :]
        self.df_validation['criteria'] = self.validation_criteria[self.frequency].values()
        results = [all(self.df_validation[col] < self.df_validation['criteria']) for col in self.df_validation.columns]
        results = ['Passed' if result == True else 'Failed' for result in results]
        results[-1] = ''
        feature_req = ['Optional' if feat not in ['energy', 'dry_temp'] else 'Important' for feat in self.df_validation.columns]
        feature_req[-1] = ''
        self.df_validation.loc['validation_result',:] = results
        self.df_validation.loc['feature_requirement',:] = feature_req
        failed_features =  self.df_validation.columns[(self.df_validation == 'Falied').any(axis=0)]
        self.data_sufficiency = True

        if not self.df_timestamps_highlights.loc['timestamp_range', 'result'] >= n_days:
            acut_range = self.df_timestamps_highlights.loc['timestamp_range', 'result']
            print(f'The data time range {acut_range} is less than ({n_days}) days. The data is insufficient.')
            self.data_sufficiency = False

        for i in range(len(self.features)):
            col_name = self.df_validation.columns[i] 
            if self.df_validation.loc['validation_result',col_name] == 'Failed':
                print("The feature ({col_name}) failed the validation test. Check the initialization summary".format(col_name=col_name))
                if self.df_validation.loc['feature_requirement',col_name] == 'Important':
                    print('The feature is required to build a valid MVBEP model. The initializer process failed')
                    self.data_sufficiency = False
                else: 
                    print('The feature is optional for building a valid MVBEP. The feature is dropped from the data frame.')
                    self.df_fin.drop(columns=[col_name], inplace=True)
                    self.features.remove(col_name)

        if self.data_sufficiency: 
            print('The initialization process is successful and the data is sufficient to build a MVBEP mode.')
            self.initializer_state = 'INITIATED'
        else:
            print('The initialization process failed and the data is insufficient to build a MVBEP model.')
            self.initializer_state == 'FAILED INITIATION'

        

    def validate_pred_data(self, 
                           data:pd.DataFrame,
                           total_missing = None, 
                           max_consec_missing = None
    ):
        # Fitting the initializer 
        self.fit(data = data,
                frequency=self.frequency,
                country_code=self.country_code,
                occupancy_schedule=self.occupancy_schedule)


        # Validation criteria
        self.validation_criteria[self.frequency]['total_missing'] = self.validation_criteria[self.frequency]['total_missing'] if total_missing is None else total_missing  
        self.validation_criteria[self.frequency]['max_consec_missing'] = self.validation_criteria[self.frequency]['max_consec_missing'] if max_consec_missing is None else max_consec_missing  

        # Timestamps validation
        self.df_fin, highlights_timestamps = self.generate_timestamp_summary(self.df_fin)

        # Other features' validation
        highlights_features = []
        for feat in self.features:
            feat_series = self.df_fin[feat]
            highlights_features.append(self.generate_feature_summary(feat_series, mark_value=True))

        # Creating highlights dataframes
        # Timestamps   
        self.df_timestamps_highlights = pd.DataFrame(
            index = highlights_timestamps.keys(), columns=['result'], data=highlights_timestamps.values()
        )

        # Other features
        indices = [list(feat.keys())[0] for feat in highlights_features]
        cols = list(highlights_features[0][indices[0]].keys())
        rows = [highlights_features[i][indices[i]] for i in range(len(indices))]
        rows = [list(row_dict.values()) for row_dict in rows]
        df_features_highlights = pd.DataFrame(index=indices, columns=cols, data=rows).transpose()

        #Checking validation requirements 
        self.df_validation = df_features_highlights.loc[['total_missing_values', 'max_consec_missing_values'], :]
        self.df_validation['criteria'] = self.validation_criteria[self.frequency].values()
        results = [all(self.df_validation[col] < self.df_validation['criteria']) for col in self.df_validation.columns]
        results = ['Passed' if result == True else 'Failed' for result in results]
        results[-1] = ''
        feature_req = ['Important' for feat in self.df_validation.columns]
        feature_req[-1] = ''
        self.df_validation.loc['validation_result',:] = results
        self.df_validation.loc['feature_requirement',:] = feature_req
        failed_features =  self.df_validation.columns[(self.df_validation == 'Falied').any(axis=0)]
        self.data_sufficiency = True


        if self.df_fin['timestamp'].min() <= self.last_date:
            print('The first timestamp in the post-retrofit data {} is before the last timestamp in the pre-retrofit data {}'.format(
                self.df_fin['timestamp'].min(), 
                self.last_date
            ))
            self.data_sufficiency = False

        for i in range(len(self.features)):
            col_name = self.df_validation.columns[i] 
            if self.df_validation.loc['validation_result',col_name] == 'Failed':
                print("The feature ({col_name}) failed the validation test. Check the initialization summary".format(col_name=col_name))
                if self.df_validation.loc['feature_requirement',col_name] == 'Important':
                    print('The feature is required for savings quantification. The initializer process failed')
                    self.data_sufficiency = False
                else: 
                    print('The feature is optional for building a valid MVBEP. The feature is dropped from the data frame.')
                    self.df_fin.drop(columns=[col_name], inplace=True)
                    self.features.remove(col_name)

        if self.data_sufficiency: 
            print('The initialization process is successful and the data is sufficient for savings quantification.')
            self.initializer_state = 'INITIATED'
        else:
            print('The initialization process failed and the data is insufficient for savings quantification.')
            self.initializer_state = 'INITIATED'

        
