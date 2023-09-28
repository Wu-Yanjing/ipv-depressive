import pickle
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import geopandas
from statsmodels.iolib.summary2 import summary_col
from statsmodels.regression.mixed_linear_model import MixedLM

from intimate_partner_violence.research_data import RawData, ResearchData, raw_data_dir, current_dir

plt.rcParams.update({'font.size': 14})
plt.rcParams['figure.constrained_layout.use'] = True

output_dir = current_dir + '/output'
raw_data_path = raw_data_dir + '/research_data_15_49_ipv.plk'
try:
    research_data = pickle.load(open(raw_data_path, 'rb'))
except FileNotFoundError:
    research_data = ResearchData()
    pickle.dump(research_data, open(raw_data_path, 'wb'))


def plot_geo_distribution(data, variable, year=2019):
    basemap = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    basemap['name'] = RawData.normalize_countries(basemap['name'])

    var = data[data['year'] == year][[research_data.country_key, variable]]
    merged_map = basemap.merge(var, left_on='name', right_on='country', how='left')
    merged_map.plot(
        column=variable,
        legend=True,
        cmap='coolwarm',
        figsize=(10, 4),
        missing_kwds=dict(color="lightgrey")
    )

    title = f"{variable.replace('_', ' ')} in {year}"
    plt.grid(False)
    plt.title(title)
    plt.savefig(output_dir + f'/{title}.png', dpi=300)
    merged_map.to_csv(output_dir + f'/{title}.csv', index=False)


if True:
    for f in [
        'IPV_indicator',
        'audiologists_and_counsellors',
        'psychologists',
        'depressive_disorders_DALYs_rate'
    ]:
        plot_geo_distribution(research_data.data, f)
