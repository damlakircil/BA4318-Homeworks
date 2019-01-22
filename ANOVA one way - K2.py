#Damla Deniz Kırçıl, 2012961
#One way ANOVA on the distance

from os import path

winabspath = "C:\\Users\\ata\Desktop"
winprojectdir = "Python"
inputfilename = "DataClear.xlsx"
inpath = path.join(winabspath, winprojectdir, inputfilename)

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import researchpy as rp
import scipy
from scipy import stats
import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

df = pd.read_excel('DataClear.xlsx', sheet_name='DataSet')

def construct(df, names, score):
    newset = [0]
    size = len(df)
    nnames = len(names)
    for index in range(1,size):
        sum = 0.0
        for name in names:
            sum = sum + float ( df[name][index]) 
        avg = sum / nnames
        diff = avg - df[score][index]
        newset.append(diff)
    return newset

names = ['İno', 'K1', 'K3', 'K4', 'İ1']
score = 'Skor'
score_diff = construct(df, names, score)
distance = df['Skor']

dfimp = pd.DataFrame({'Skor':score_diff, 'K2':distance})
summary = rp.summary_cont(dfimp['Skor'].groupby(dfimp['K2']))
print (summary)
corr = dfimp.corr()
print(corr)
y = dfimp['Skor']
x = dfimp['K2']
plt.scatter(x, y)
plt.show()

few = 30
dfimp_last = dfimp[-few-1:-1]
summary2 = rp.summary_cont(dfimp_last['Skor'].groupby(dfimp_last['K2']))
print (summary2)
corr2 = dfimp_last.corr()
print(corr2)
y = dfimp_last['Skor']
x = dfimp_last['K2']
plt.scatter(x, y)
plt.show()

print("By conducting one way ANOVA,")
set1 = dfimp['Skor'][ dfimp['K2'] == 0 ]
set2 = dfimp['Skor'][ dfimp['K2'] == 1 ]
F, p = stats.f_oneway(set1, set2)
print(F, p)
plevel = 0.01
Flevel = 6.4149
if p < plevel and F > Flevel :
    print("p-value less than ", plevel)
    print("F value is larger than ", Flevel)
    print("Variance is explained by the distance factor.")
else:
    print("Variance is not explained by the distance factor.")