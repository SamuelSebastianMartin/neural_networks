#! /usr/bin/env python3

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv('regression/owid-covid-data.csv')
plt.scatter(df['new_tests_per_thousand'], df['new_cases_per_million'])
plt.xlabel('New tests per 1000')
plt.ylabel('New Cases per million')
plt.title('Corona virus testing, by country')
plt.show()
"""
'location', 'date', 'total_cases', 'new_cases',
       'total_deaths', 'new_deaths', 'total_cases_per_million',
       'new_cases_per_million', 'total_deaths_per_million',
       'new_deaths_per_million', 'total_tests', 'new_tests',
       'total_tests_per_thousand', 'new_tests_per_thousand',
"""
