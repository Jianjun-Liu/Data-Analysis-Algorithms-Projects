# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:59:05 2020

@author: CUP
"""

import matplotlib.pyplot as plt #用于画图工具
plt.rcParams['font.sans-serif'] = ['SimHei']#SimHei是黑体的意思
plt.rcParams['axes.unicode_minus'] = False#avoid negtive symbol
import seaborn as sns
import pandas as pd

#from efficient_apriori import apriori
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

file ="Online Retail_Small.xlsx"

df = pd.read_excel('Online Retail_Small.xlsx')
df.head()
df.info()
#%%
df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]

df = df[~df['Description'].str.contains('POSTAGE')]
#%%
basket = (df[df['Country'] =="France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

#%%
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
#basket_sets.drop('POSTAGE', inplace=True, axis=1)

#%%
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
#itemsets, rules = apriori(basket_sets, min_support=0.5,min_confidence=1)
rules.head()

rules[ (rules['lift'] >= 6) & (rules['confidence'] >= 0.8) ]

basket['ALARM CLOCK BAKELIKE GREEN'].sum()

basket['ALARM CLOCK BAKELIKE RED'].sum()

basket2 = (df[df['Country'] =="Germany"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

basket_sets2 = basket2.applymap(encode_units)
basket_sets2.drop('POSTAGE', inplace=True, axis=1)
frequent_itemsets2 = apriori(basket_sets2, min_support=0.05, use_colnames=True)
rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)

rules2[ (rules2['lift'] >= 4) & (rules2['confidence'] >= 0.5)]
