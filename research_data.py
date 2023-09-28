from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from os.path import dirname, realpath
from typing import Literal
import country_converter
import pickle

current_dir = dirname(realpath(__file__))
raw_data_dir = current_dir + '/raw_data'


@dataclass
class RawData:
    country_key = 'country'
    year_key = 'year'
    keys = [country_key, year_key]

    @abstractmethod
    def load(self, *args, **kwargs):
        ...

    def __post_init__(self):
        data = self.load()
        data[self.country_key] = self.normalize_countries(data[self.country_key])
        self.data = data

    @staticmethod
    def normalize_countries(raw_countries: pd.Series):
        unique_countries = raw_countries.unique()

        normalized_countries = country_converter.convert(
            unique_countries, to='name_short', not_found=None
        )

        reference_dict = {
            unique_countries[i]: normalized_countries[i]
            for i in range(len(unique_countries))
        }

        return raw_countries.apply(lambda c: reference_dict.get(c))


"""
GBD Results tool:
Use the following to cite data included in this download:
Global Burden of Disease Collaborative Network.
Global Burden of Disease Study 2019 (GBD 2019) Results.
Seattle, United States: Institute for Health Metrics and Evaluation (IHME), 2020.
Available from https://vizhub.healthdata.org/gbd-results/.

"""


@dataclass
class MentalHealthData(RawData):
    cause: Literal['mental disorder', 'anxiety disorder', 'depressive disorder'] = None

    def load(self):
        raw_data = pd.read_csv(raw_data_dir + '/IHME-GBD_2019_GBD_15_49_ipv.csv')
        raw_data['measure_name'] = raw_data['measure_name'].apply(lambda a: a.split(' ')[0])
        raw_data['cause_name'] = raw_data['cause_name'].apply(lambda a: a.lower().replace(' ', '_'))

        processed_data = pd.DataFrame()
        for measure_name in raw_data['measure_name'].unique():
            for cause_name in raw_data['cause_name'].unique():

                if self.cause:
                    if self.cause not in cause_name:
                        continue

                temp_data = \
                    raw_data[
                        (raw_data.measure_name == measure_name) &
                        (raw_data.cause_name == cause_name)
                        ][['location_name', 'year', 'val']].rename(
                        columns={
                            'location_name': self.country_key,
                            'val': f'{cause_name}_{measure_name}_rate'
                        }
                    )

                if temp_data.empty:
                    continue

                if processed_data.empty:
                    processed_data = temp_data
                else:
                    processed_data = pd.merge(
                        processed_data,
                        temp_data,
                        on=self.keys
                    )

        return processed_data

    def __post_init__(self):
        super().__post_init__()


class IPVData(RawData):

    def load(self):
        raw_data = pd.read_csv(raw_data_dir + '/IHME_IPV_1990_2021.csv')
        rename_dict = {
            'location_name': self.country_key,
            'year_id': self.year_key,
            'value': 'IPV_indicator'
        }

        processed_data = raw_data[rename_dict.keys()].rename(columns=rename_dict)

        n_group = 3
        processed_data['IPV_category'] = processed_data.groupby('year')['IPV_indicator'].transform(
            lambda df: pd.qcut(
                df,
                q=np.arange(0, 1 + 1 / n_group, 1 / n_group),
                labels=['low IPV', 'middle IPV', 'high IPV']
            )
        )

        return processed_data

    def __post_init__(self):
        super().__post_init__()


class SDIData(RawData):

    def load(self, *args, **kwargs):
        raw_data = pd.read_csv(raw_data_dir + '/IHME-GBD_2019_SDI.csv', encoding_errors='ignore')

        processed_data = pd.DataFrame()
        for idx in np.arange(len(raw_data)):
            row = raw_data.loc[idx, :].reset_index()

            c = row.iloc[0, 1]

            temp_data = pd.DataFrame(
                dict(
                    year=row.iloc[1:, 0].astype(int),
                    socio_demographic_index=row.iloc[1:, 1].astype(float)
                ),
            )
            temp_data[self.country_key] = c
            processed_data = pd.concat([processed_data, temp_data], ignore_index=True)

        processed_data['SDI_category'] = processed_data.groupby('year')['socio_demographic_index'].transform(
            lambda df: pd.qcut(
                df,
                q=[0, 0.2, 0.4, 0.6, 0.8, 1],
                labels=['low SDI', 'low-middel SDI', 'middle SDI', 'middle-high SDI', 'high SDI'],
            )
        )

        return processed_data

    def __post_init__(self):
        super().__post_init__()


class HealthWorkerData(RawData):

    def load(self, *args, **kwargs):
        raw_data = pd.read_csv(raw_data_dir + '/IHME-GBD_2019_Health_Worker_Density.csv')

        processed_data = pd.DataFrame()
        for cate in raw_data.cadre.unique():
            rename_dict = {
                'location_name': self.country_key,
                'year_id': self.year_key,
                'mean': cate.lower().replace(' & ', ' ').replace(' ', '_').replace(',', '')
            }

            temp = raw_data[raw_data.cadre == cate][rename_dict.keys()].rename(
                columns=rename_dict
            )

            if processed_data.empty:
                processed_data = temp
            else:
                processed_data = pd.merge(processed_data, temp, on=self.keys)

        return processed_data

    def __post_init__(self):
        super().__post_init__()


class SocialExpenditure(RawData):

    def load(self):
        raw_data = pd.read_csv(raw_data_dir + '/OECD_Social_Expenditure.csv')
        raw_data = raw_data[
            (raw_data['Measure'] == 'In percentage of Gross Domestic Product') &
            (raw_data['Type of Programme'] == 'Total') &
            (raw_data['Type of Expenditure'] == 'Total') &
            (raw_data['Source'] == 'Public')
            ]

        processed_data = pd.DataFrame()
        for branch in raw_data['Branch'].unique():

            branch_data = raw_data[raw_data['Branch'] == branch]

            rename_dict = {
                'Year': self.year_key,
                'Country': self.country_key,
                'Value': branch.lower() + '_social_expenditure'
            }

            temp = branch_data[rename_dict.keys()].rename(columns=rename_dict)

            if processed_data.empty:
                processed_data = temp
            else:
                processed_data = pd.merge(processed_data, temp, on=self.keys, how='outer')

        return processed_data

    def __post_init__(self):
        super().__post_init__()


@dataclass
class ResearchData:
    cause: Literal['mental disorder', 'anxiety disorder', 'depressive disorder'] = None

    country_key = 'country'
    year_key = 'year'
    keys = [country_key, year_key]

    def __post_init__(self):

        self.mental_health_data = MentalHealthData(cause=self.cause).data
        self.ipv_data = IPVData().data
        self.sdi_data = SDIData().data
        self.health_worker_data = HealthWorkerData().data

        self.data = self.auto_merge()

    def auto_merge(self):

        data = pd.DataFrame()
        for attr, df in self.__dict__.items():
            if isinstance(df, pd.DataFrame):

                self.__setattr__(attr, df)

                if data.empty:
                    data = df
                else:
                    data = pd.merge(data, df, on=self.keys)

        return data

    @property
    def years(self):
        return self.data[self.country_key].unique()

    @property
    def countries(self):
        return self.data[self.country_key].unique()

    @property
    def ipv_categories(self):
        return self.data['IPV_category'].sort_values().unique()

    @property
    def sdi_categories(self):
        return self.data['SDI_category'].sort_values().unique()
