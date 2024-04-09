# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:52:44 2020

@author: benan
"""
import numpy as np
import pandas as pd


def isNaN(num):
    return num != num
#%%

class LambdaDataset:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def lambda_pre_process(self, dataframe, optic):
            optic_first = dataframe.columns.get_loc(optic + "_0")
            optic_last  =dataframe.columns.get_loc(optic + "_444")+1
            optic_list = []

            for i,j in enumerate(dataframe.index): # row boyunca döngüye sokar her satır numune için yukarıdaki metodu uygular
                optic_list = optic_list + list(dataframe.iloc[i,optic_first:optic_last].values)

            optic_df = pd.DataFrame(optic_list)
            optic_df.columns = [optic]
            return optic_df


    def create_lambda_dataset(self):
        dataframe = self.dataframe
        column_name_list = ["T_pre", "Rc_pre", "Ru_pre", "T_post", "Rc_post", "Ru_post"]

        sample_size = len(list(dataframe.index))

        df_list = []
        df_lambda = pd.DataFrame(sample_size * list(np.arange(280,2505,5)))
        df_lambda.columns = ["lambda"]
        df_list.append(df_lambda)
        for i,optic in enumerate(column_name_list):
            temp = self.lambda_pre_process(dataframe, optic)
            df_list.append(temp)

        lambda_dataset = pd.concat(df_list, axis =1)
        return lambda_dataset












































































