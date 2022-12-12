from scipy import stats
from sklearn import preprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm

def P_values(N_data,T_data,feature):
    print("Generating P_values feature...")
    t = {}
    comm = [i for i in feature['GeneName'] if i in N_data.index]
    feature = feature[feature['GeneName'].isin(comm)]
    N_data = N_data.loc[comm,:]
    T_data = T_data.loc[comm, :]
    normal_mean = N_data.mean(axis=1)
    tumor_mean = T_data.mean(axis=1)
    fold = (tumor_mean) / (normal_mean)
    fold = fold.replace([np.inf, -np.inf], np.nan)
    fold = fold.fillna(0)
    for i in tqdm(N_data.index):

        tmp = N_data.loc[i, :].values.tolist()
        for j in range(len(N_data.loc[i, :]), len(T_data.loc[i, :])):
            tmp.append(N_data.loc[i, :].values.mean() + np.random.randn())
        ttestup = stats.wilcoxon(tmp, T_data.loc[i, :], alternative='greater')
        ttestdo = stats.wilcoxon(tmp, T_data.loc[i, :], alternative='less')

        if ttestup[1] < 0.05:
            t[i]=1
            continue

        if ttestdo[1] < 0.05:
            t[i]=-1
            continue

        if fold[i] > 1.5:
            t[i]=-1
            continue

        else:

            t[i]=0
    p_values_series=pd.Series(data=t.values(),index=t.keys())
    # comm = [i for i in feature['GeneName'] if i in p_values_series.index]
    # feature = feature[feature['GeneName'].isin(comm)]
    # p_values_series = p_values_series[comm]
    feature["p_values"] = preprocessing.scale(p_values_series.values)
    print(feature)

    print("Generate P_values feature complete")
    return feature
