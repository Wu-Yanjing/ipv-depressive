import pickle
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import geopandas

from intimate_partner_violence.research_data import ResearchData, raw_data_dir, current_dir

plt.rcParams.update({'font.size': 14})
plt.rcParams['figure.constrained_layout.use'] = True

output_dir = current_dir + '/output'
raw_data_path = raw_data_dir + '/research_data.plk'
try:
    research_data = pickle.load(open(raw_data_path, 'rb'))
except:
    research_data = ResearchData()
    pickle.dump(research_data, open(raw_data_path, 'wb'))

full_data = research_data.data

from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.iolib.summary2 import summary_col


# disease = 'mental_disorders'

def reg(
        countries_group=None,
        disease: Literal['mental', 'depressive', 'anxiety'] = None
):
    X_cols = [
        'year',
        'IPV',
        'socio_demographic_index',
        'psychologists',
        'audiologists_and_counsellors',
        'interation',
    ]

    if countries_group:
        group = research_data.countries_categories[countries_group]
        group_data = full_data[full_data[research_data.country_key].isin(group)].copy()

    else:
        group_data = full_data.copy()

    group_data['audiologists_and_counsellors'] = group_data.groupby(
        research_data.country_key
    )['audiologists_and_counsellors'].shift(1)
    group_data['psychologists'] = group_data.groupby(
        research_data.country_key
    )['psychologists'].shift(1)

    group_data = group_data.dropna()

    group_data['interation'] = (
                                       group_data['psychologists'] + group_data['audiologists_and_counsellors']
                               ) * group_data['IPV']

    for y_col in research_data.burden_cols:

        if ('YLLs' in y_col) or (disease not in y_col):
            continue

        temp_y = group_data[y_col].rename(y_col.replace('_', ' '))
        temp_X = group_data[X_cols].rename(columns={
            original: original.replace('_', ' ')
            for original in X_cols
        })

        model = MixedLM(
            endog=temp_y,
            exog=temp_X,
            groups=group_data[research_data.country_key],
        ).fit(reml=True)

        print(model.summary())


# for group in research_data.countries_categories.keys():
#     print(group)
#
#     reg(group, disease='mental')

# reg(disease='depressive')


def seg_reg(
        group_data,
        disease: Literal['mental', 'depressive', 'anxiety'] = None
):
    X_cols = [
        'year',
        'IPV_indicator',
        'socio_demographic_index',
        'psychologists',
        'audiologists_and_counsellors',
        'interation',
    ]

    group_data['audiologists_and_counsellors'] = group_data.groupby(
        research_data.country_key
    )['audiologists_and_counsellors'].shift(1)
    group_data['psychologists'] = group_data.groupby(
        research_data.country_key
    )['psychologists'].shift(1)

    group_data = group_data.dropna()

    group_data['interation'] = (
                                       group_data['psychologists'] + group_data['audiologists_and_counsellors']
                               ) * group_data['IPV']

    for y_col in research_data.burden_cols:

        if ('YLLs' in y_col) or (disease not in y_col):
            continue

        temp_y = group_data[y_col].rename(y_col.replace('_', ' '))
        temp_X = group_data[X_cols].rename(columns={
            original: original.replace('_', ' ')
            for original in X_cols
        })

        model = MixedLM(
            endog=temp_y,
            exog=temp_X,
            groups=group_data[research_data.country_key],
        ).fit(reml=True)

        print(model.summary())


full_data['cat'] = pd.qcut(full_data['IPV'], q=[0, 0.25, 0.5, 0.75, 1])

for c in full_data['cat'].unique():
    g = full_data[full_data['cat'] == c]
    seg_reg(g, 'mental')

print()

import piecewise_regression

X_cols = [
    'year',
    'socio_demographic_index',
    'psychologists',
    'audiologists_and_counsellors'
]
# ms = piecewise_regression.ModelSelection(
#     full_data['IPV'].tolist(),
#     full_data['depressive_disorders_Prevalence_rate'].tolist()
# )
# pw_fit = piecewise_regression.Fit(
#     (full_data['IPV'] * full_data['psychologists']).tolist(),
#     full_data['depressive_disorders_Prevalence_rate'].tolist(), n_breakpoints=2
# )
# # Plot the data, fit, breakpoints and confidence intervals
# pw_fit.plot_data(color="grey", s=20)
# # Pass in standard matplotlib keywords to control any of the plots
# pw_fit.plot_fit(color="red", linewidth=4)
# pw_fit.plot_breakpoints()
# pw_fit.plot_breakpoint_confidence_intervals()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

if False:
    full_data = research_data.data

    n_group = 4
    full_data['IPV_category'] = full_data.groupby(research_data.year_key)['IPV'].transform(
        lambda df: pd.qcut(
            df,
            q=np.arange(0, 1 + 1 / n_group, 1 / n_group),
            labels=['low IPV', 'low-middle IPV', 'middle-high IPV', 'high IPV']
        )
    )


    def export_regression_results(
            data,
            disease: Literal['mental', 'depressive', 'anxiety']
    ):
        group = data['IPV_category'].iloc[0]
        print(group)

        data['IPV * socio_demographic_index'] = data['IPV'] * data['socio_demographic_index']

        X_cols = [
            'year',
            'IPV',
            'audiologists_and_counsellors',
            'IPV * socio_demographic_index',
            'psychologists',
            'socio_demographic_index'
        ]

        output = []
        for y_col in research_data.burden_cols:

            if ('YLLs' in y_col) or (disease not in y_col):
                continue

            temp_y = data[y_col].rename(y_col.replace('_', ' '))
            temp_X = data[X_cols].rename(columns={
                original: original.replace('_', ' ')
                for original in X_cols
            })

            model = MixedLM(
                endog=temp_y,
                exog=temp_X,
                groups=data[research_data.country_key],
            ).fit()

            output += [model]

        if len(output) > 0:
            results_table = summary_col(
                output,
                stars=True,

            )

            results_table.tables[0].iloc[::2].to_csv(
                output_dir +
                f"/regression results {disease} {group.replace('_', ' ')}.csv"
            )

            print(
                results_table
            )


    full_data.groupby('IPV_category').apply(export_regression_results, disease='depressive')
    full_data.groupby('IPV_category').apply(export_regression_results, disease='mental')