import pickle
from dataclasses import dataclass, field
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

from linearmodels.panel import PanelOLS, RandomEffects
from linearmodels.panel.results import PanelEffectsResults, RandomEffectsResults
from linearmodels.panel.results import PanelModelComparison

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

# Hausman test
import numpy.linalg as la
from scipy import stats


def hausman(fe, re):
    b = fe.params.iloc[1:]
    B = re.params.iloc[1:]
    v_b = fe.cov.iloc[1:, 1:]
    v_B = re.cov.iloc[1:, 1:]

    df = b[np.abs(b) < 1e8].size
    chi2 = np.dot((b - B).T, la.inv(v_b - v_B).dot(b - B))

    pval = stats.chi2.sf(chi2, df)
    return chi2, df, pval


@dataclass
class AssumptionTestResults:
    Test: Literal['Breusch-Pagan Test', 'Durbin-Watson Test', 'Hausman-Test'] = None
    H0: Optional[str] = None
    H1: Optional[str] = None
    stat: Optional[float] = None
    pvalue: Optional[float] = None
    conclusion: str = field(init=False, default=None)

    def evaluate(self):
        if self.pvalue <= 0.05:
            self.conclusion = self.H1
        else:
            self.conclusion = self.H0

    def __from_dict__(self, attr_dict: dict):
        for k, v in attr_dict.items():
            self.__setattr__(
                k, v
            )

    def __post_init__(self):
        self.evaluate()


@dataclass
class BreuschPaganTestResults(AssumptionTestResults):

    def __post_init__(self):
        attrs = dict(
            Test='Breusch-Pagan Test',
            H0='Homoskedasticity',
            H1='Heteroskedasticity'
        )
        self.__from_dict__(attrs)
        super().__post_init__()


@dataclass
class DurbinWatsonTestResults(AssumptionTestResults):

    def evaluate(self):
        if 0 < self.stat < 2:
            self.conclusion = 'positive auto correlation'
        elif 2 < self.stat < 4:
            self.conclusion = 'negative auto correlation'
        else:
            self.conclusion = 'no auto correlation'

    def __post_init__(self):

        attrs = dict(
            Test='Durbin-Watson Test',
            H0='Non-Auto-correlation',
            H1='Auto-correlation'
        )
        self.__from_dict__(attrs)
        super().__post_init__()


@dataclass
class HausmanTestResults(AssumptionTestResults):

    def __post_init__(self):
        attrs = dict(
            Test='Hausman-Test',
            H0='random effect',
            H1='fixed effect'
        )
        self.__from_dict__(attrs)
        super().__post_init__()


@dataclass
class PanelRegResultsWrapper:
    pols: PanelEffectsResults
    fe: PanelEffectsResults
    re: RandomEffectsResults

    @property
    def assumption_test(self):
        # Breusch-Pagan LM test
        from statsmodels.stats.diagnostic import het_breuschpagan
        lm_stat, lm_pvalue, _, _ = het_breuschpagan(self.pols.resids, self.pols.model.exog.dataframe)
        bpt = BreuschPaganTestResults(
            stat=lm_stat,
            pvalue=lm_pvalue
        )

        # Durbin-Watson-Test
        from statsmodels.stats.stattools import durbin_watson
        dw_stat = durbin_watson(self.pols.resids)
        dwt = DurbinWatsonTestResults(
            stat=dw_stat
        )

        # Hausman-Test
        hausman_stat, _, hausman_pvalue = hausman(self.fe, self.re)
        ht = HausmanTestResults(
            stat=hausman_stat,
            pvalue=hausman_pvalue
        )

        res = pd.DataFrame([bpt, dwt, ht]).transpose()

        return res

    @property
    def export_regression_results(self):
        results_table = PanelModelComparison(
            [self.pols, self.fe, self.re],
            stars=True
        )

        res = pd.DataFrame(
            results_table.summary.tables[0]
        )

        return res


full_data = research_data.data.copy()


def segment_reg(data, disease):
    y_col = {
        'depressive': 'depressive_disorders_DALYs_rate',
        'mental': 'mental_disorders_DALYs_rate'
    }.get(disease, lambda a: a)

    data = data.set_index(keys=['country', 'year']).dropna()

    X_cols = [
        'IPV_indicator',
        'audiologists_and_counsellors',
        'psychologists',
        'physicians_clinical_officers_and_chws',
        'aides_emergency_medical_workers',
        'nursing_and_midwifery_personnel',
        'all_health_workers',
        'socio_demographic_index'
    ]

    cate = []
    at_res = pd.DataFrame()
    re_res = []
    fe_res = []
    pols_res = []
    for idx, ipv_group in enumerate(research_data.ipv_categories):
        seg = data[data['IPV_category'] == ipv_group]

        y = seg[y_col]
        X = sm.add_constant(seg[X_cols])

        pols = PanelOLS(y, X).fit()
        fe = PanelOLS(y, X, entity_effects=True, time_effects=True).fit()
        re = RandomEffects(y, X).fit()

        at = PanelRegResultsWrapper(
            pols=pols,
            fe=fe,
            re=re
        ).assumption_test

        for i in ['stat', 'pvalue']:
            at.loc[i] = at.loc[i].apply(lambda a: round(float(a), 4))
        at.insert(0, 'ipv_category', ipv_group)
        at_res = pd.concat([at_res, at])

        re_res += [re]
        fe_res += [fe]
        pols_res += [pols]
        cate += [ipv_group]

    for model_name, model_res in zip(
            ['panel ols', 'fixed effect', 'random effect'],
            [pols_res, fe_res, re_res]
    ):
        results_table = PanelModelComparison(
            model_res,
            stars=True
        )

        res = pd.DataFrame(
            results_table.summary.tables[0]
        )

        res.iloc[0, 1:] = cate

        res.iloc[1::2] = res.iloc[1::2].astype(str)

        res.to_csv(output_dir + f'/regression_{model_name}_{disease}.csv', index=False)
        at_res.to_csv(output_dir + f'/assumption_test_{disease}.csv')


segment_reg(full_data, 'depressive')
print()
