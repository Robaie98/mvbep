import pandas as pd
import numpy as np
import datetime
import statsmodels.formula.api as smf 



class TOWT_Transformer:
    def __init__(self):
        self.df_fin = None
        self.timestamp_col = None
        self.energy_col = None
        self.temp_col = None
        self.segments_count = None
        self.residuals_threshold = None
        self.design_matrix_features = []
        self.transformer_state = None

    def fit(
        self,
        df_input:pd.DataFrame,
        timestamp_col_name:str,
        energy_col_name:str,
        temp_col_name:str,
        cdd_temp = 65,
        hdd_temp = 55,
        segments_count = 6,
        threshold = 0.35
    ):
        self.df_fin = df_input.copy()
        self.timestamp_col = timestamp_col_name
        self.energy_col = energy_col_name
        self.temp_col = temp_col_name
        self.cdd_temp = cdd_temp
        self.hdd_temp = hdd_temp
        self.segments_count = segments_count
        self.threshold = threshold

    def create_preliminary_features(self, df_input:pd.DataFrame, cdd_temp:int, hdd_temp:int):
        df = df_input.copy()
        df['hour_of_week'] = df['day_of_week']*24 + df['hour_of_day']
        df[f'cdd_{cdd_temp}'] = df[self.temp_col].apply(lambda x: max(x - cdd_temp, 0)) 
        df[f'hdd_{hdd_temp}'] = df[self.temp_col].apply(lambda x: max(hdd_temp - x, 0))
        self.design_matrix_features.append('hour_of_week')
        return df

    def create_temp_segments(self, df_input:pd.DataFrame, segments_count:int):
        # Creating Segments of equal width 
        df = df_input.copy()
        min_temp = min(df[self.temp_col]); max_temp = max(df[self.temp_col]); 
        segment_width = (max_temp - min_temp) / segments_count
        lower_bounds = [value for value in min_temp + segment_width*np.array(range(segments_count))]
        upper_bounds = np.flip([value for value in max_temp - segment_width*np.array(range(segments_count))])
        segments_bounds = zip(lower_bounds, upper_bounds)
        # Creating columns in df representing segmented
        segment_names = []
        i = 1
        for lwr,upr in segments_bounds:
            df['temp_seg_{}'.format(i)] = df[self.temp_col].apply(lambda x: max(0, x-lwr))
            df['temp_seg_{}'.format(i)] = df['temp_seg_{}'.format(i)].apply(lambda x: min(segment_width, x))
            segment_names.append('temp_seg_{}'.format(i))
            i += 1
        return df, segment_names

    def estimate_occupancy(self, df_input:pd.DataFrame, threshold:float, cdd_temp:int, hdd_temp:int):
        df = df_input.copy()
        usage_model = smf.wls(formula="{} ~ cdd_{} + hdd_{}".format(self.energy_col, cdd_temp, hdd_temp), data=df)
        df['residuals'] = usage_model.fit().resid
        df['is_positive'] = df['residuals'] > 0
        positive_residuals_count = df.groupby(['hour_of_week'])['is_positive'].sum()
        residuals_count = df.groupby(['hour_of_week'])['residuals'].count()
        ratios = positive_residuals_count / residuals_count
        occupancy_state = (ratios > threshold).astype(int) 
        df = df.merge(occupancy_state.to_frame('occupancy'), on = 'hour_of_week')
        return df

    def prepare_design_matrix(self,
        df_input:pd.DataFrame = None,
    ):
        df_output = self.create_preliminary_features(
            df_input = df_input if df_input is not None else self.df_fin,
            cdd_temp = self.cdd_temp,
            hdd_temp = self.hdd_temp
        )
        
        

        df_output, segment_names = self.create_temp_segments(
                                df_input = df_output,
                                segments_count = self.segments_count
                                )
        self.design_matrix_features.extend(segment_names)
        df_output = self.estimate_occupancy(
            df_input = df_output,
            threshold = self.threshold,
            cdd_temp = self.cdd_temp,
            hdd_temp = self.hdd_temp)
        self.design_matrix_features.append('occupancy')

        return df_output, self.design_matrix_features