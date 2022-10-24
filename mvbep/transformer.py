import pandas as pd
import numpy as np 
from datetime import datetime
import holidays
from .towt_utils import TOWT_Transformer 
import re

class Transformer:

    def __init__(
        self
    ):
        self.df_init = None
        self.df_fin = None
        self.timestamp_frequency = None
        self.time_features = None
        self.occupancy_schedule = None
        self.country_code = None
        self.downsample = None
        self.design_matrix_features = None
        self.towt_design_matrix = None
        self.towt_design_matrix_features = None
        self.trans_scaler = None


    def fit(
        self,
        data: pd.DataFrame,
        timestamp_frequency: str,
        optional_features:list = None,
        occupancy_schedule: dict = None,
        country_code: str = None,
        downsample_from_to: str = None
    ):
        time_features = [
            'hour_of_day',
            'hour_of_week',
            'day_of_week',
            'day_of_month',
            'day_of_year'
        ]
        df_init = data.copy()
        self.optional_features = optional_features
        self.timestamp_frequency = timestamp_frequency
        if timestamp_frequency == '15-min':
            self.time_features = time_features
            self.time_features.append('minute_of_hour')
        elif timestamp_frequency == 'hourly':
            self.time_features = time_features
        elif timestamp_frequency == 'daily':
            self.time_features = [feat for feat in time_features if feat not in ['hour_of_day', 'hour_of_week']]
        else:
            raise ValueError(f"The passed timestamp frequency '{timestamp_frequency}' is not valid.")
        self.occupancy_schedule = occupancy_schedule
        self.country_code = country_code
        downsample_from_to = downsample_from_to 

        if downsample_from_to is not None:
            self.df_fin = self.downsample_frequency(df_input = df_init, downsample_from_to = downsample_from_to)
        else:
            self.df_fin = df_init


    def creat_cyclic_features(self, series:pd.Series):
        max_value = series.max()
        sin_values = [np.sin((2*np.pi*x)/max_value) for x in list(series)]
        cos_values = [np.cos((2*np.pi*x)/max_value) for x in list(series)]
        return sin_values, cos_values

    def create_time_features(self, df_input:pd.DataFrame):
        # Returns several columns that are in cyclic_features and self.time_features
        df = df_input.copy()
        dt_attributes = {
        'minute_of_hour':'minute',
        'hour_of_day':'hour',
        'day_of_week':'weekday',
        'day_of_month':'day',
        'day_of_year':'dayofyear'
        }
        cyclic_features = []
        for feat, att in dt_attributes.items():
            df[feat] = getattr(df['timestamp'].dt, att)
        if self.timestamp_frequency != 'daily':
            df['hour_of_week'] = df['day_of_week']*24 + df['hour_of_day']
        df.drop(columns=[feat for feat in dt_attributes.keys() if feat not in self.time_features], inplace=True)
        for feat in self.time_features:
            df[feat+'_sin'], df[feat+'_cos'] = self.creat_cyclic_features(df[feat])
            cyclic_features.extend([feat+'_sin', feat+'_cos'])
        return df, cyclic_features

    def create_occupany_schedule_features(self, df_input:pd.DataFrame):
        # Returns a dataframe with an additional column called schedule
        df = df_input.copy()
        count = df.shape[0]
        df['schedule'] = 0
        # Weekly
        if self.timestamp_frequency != 'daily':
            for week_day in self.occupancy_schedule['weekly'].keys():
                if self.occupancy_schedule['weekly'][week_day] is not None:
                    for sched_range, occupancy_ratio in self.occupancy_schedule['weekly'][week_day].items():
                        first_bound, last_bound = sched_range.split('-')
                        first_bound, last_bound = int(first_bound), int(last_bound)
                        unoccupied_hours = list(range(first_bound, last_bound)) 
                        mask = (df['day_of_week'] == week_day) & (df['hour_of_day'].isin(unoccupied_hours))
                        df.loc[mask,'schedule'] = occupancy_ratio
        # Annual
        if self.occupancy_schedule['annually'] is not None:
            for day in self.occupancy_schedule['annually']:
                day_of_year = datetime.strptime(day, '%m-%d').timetuple().tm_yday
                mask = df['day_of_year']==day_of_year
                df.loc[mask,'schedule'] = 1

        return df

    def create_holiday_feature(self, df_input:pd.DataFrame):
        # Returns a dataframe with additional columns that are in holiday_names
        df = df_input.copy()
        years = df['timestamp'].dt.year.unique().tolist()
        holiday_dict = holidays.country_holidays(self.country_code, years=years)
        df_holid = pd.DataFrame(data = zip(holiday_dict.keys(), holiday_dict.values()), columns=['dates', 'names'])
        df_holid = pd.DataFrame(df_holid.reset_index().groupby(['names'])['dates'].apply(list)).reset_index()
        holiday_list = df_holid.to_dict(orient='records')
        holiday_names = []
        for holiday in holiday_list:
            holiday_names.append(holiday['names'])
            df[holiday['names']] = 0
            for date in holiday['dates']:
                mask = df['timestamp'].dt.date == date
                df.loc[mask, holiday['names']] = 1
        return df, holiday_names

    def cdd_temp(self, df_input:pd.DataFrame, cdd_temp:int = 65, hdd_temp:int = 55):
        df = df_input.copy()
        df[f'cdd_{cdd_temp}'] = df['dry_temp'].apply(lambda x: max(x - cdd_temp, 0))
        df[f'hdd_{hdd_temp}'] = df['dry_temp'].apply(lambda x: max(hdd_temp - x, 0))
        return df, [f'cdd_{cdd_temp}', f'hdd_{hdd_temp}']

    def downsample_frequency(self, df_input:pd.DataFrame, downsample_from_to:str):
        df = df_input.copy()
        mean_features = [feat for feat in df.columns if feat not in ['timestamp', 'energy']]

        # 15-min to hourly frequency conversion
        if downsample_from_to == '15-min~hourly':
            df['timestamp'] = df['timestamp'].dt.floor('H') 
            first_group = df.groupby(['timestamp'])[['energy']].sum()
            second_group = df.groupby(['timestamp'])[mean_features].mean()
            df = first_group.merge(second_group, on='timestamp')
            df.reset_index(inplace=True)
            self.fit(df,
                    timestamp_frequency = 'hourly', 
                    optional_features = self.optional_features,
                    occupancy_schedule = self.occupancy_schedule,
                    country_code = self.country_code,
                    downsample_from_to = None)

        # hourly to daily frequency conversion
        elif downsample_from_to == 'hourly~daily' or downsample_from_to == '15-min~daily':
            df['timestamp'] = df['timestamp'].dt.floor('d')
            df, temp_dd_features = self.cdd_temp(df_input=df)
            first_group = df.groupby(['timestamp'])[['energy']+temp_dd_features].sum()
            second_group = df.groupby(['timestamp'])[mean_features].mean()
            df = first_group.merge(second_group, on='timestamp')
            df.reset_index(inplace=True)
            self.fit(df,
                    timestamp_frequency = 'daily',
                    optional_features = self.optional_features,
                    occupancy_schedule = self.occupancy_schedule,
                    country_code = self.country_code,
                    downsample_from_to = None)
        else:
            raise ValueError(f"The passed from_to value '{downsample_from_to}' is invalid.")
        return df


    def transform(self):
        self.design_matrix_features = [] if self.optional_features is None else self.optional_features
        self.design_matrix_features = [feat for feat in self.design_matrix_features if feat not in ['dry_temp', 'energy']]
        self.df_fin.sort_values('timestamp', inplace=True)

        # Creating cyclic features
        self.df_fin, cyclic_features = self.create_time_features(self.df_fin)
        self.design_matrix_features.extend(cyclic_features)

        # 15-min case
        if self.timestamp_frequency == '15-min':
            self.design_matrix_features.extend(['energy', 'dry_temp'])
        
        # Hourly case
        elif self.timestamp_frequency == 'hourly':
            self.design_matrix_features.extend(['energy', 'dry_temp'])
            towt_transformer = TOWT_Transformer()
            towt_transformer.fit(
                df_input = self.df_fin, 
                timestamp_col_name = 'timestamp',
                energy_col_name = 'energy',
                temp_col_name = 'dry_temp'
            )
            self.towt_design_matrix, self.towt_design_matrix_features = towt_transformer.prepare_design_matrix(
                df_input = self.df_fin
            )
            self.towt_design_matrix.sort_values('timestamp', inplace=True)
        
        # Daily case
        elif self.timestamp_frequency == 'daily':
            self.design_matrix_features.extend(['energy', 'cdd_65', 'hdd_55'])
        
        # Creating schedule feature
        if self.occupancy_schedule is not None:
            self.df_fin = self.create_occupany_schedule_features(self.df_fin)
            self.design_matrix_features.append('schedule')
        
        # Creating holiday features 
        if self.country_code is not None:
            try:
                self.df_fin, holiday_features = self.create_holiday_feature(self.df_fin)
                self.design_matrix_features.extend(holiday_features)
            except:
                print('Holiday features generation process failed')

        # Changing features' names in case names included special characters from holidays package
        self.design_matrix_features.remove('energy')
        redefined_features = [re.sub('[^A-Za-z0-9]+', '_', feat) for feat in self.design_matrix_features]
        self.df_fin.rename(columns={feat:new_feat for feat, new_feat in zip(
            self.design_matrix_features, redefined_features)},
            inplace=True)
        self.design_matrix_features = redefined_features
        