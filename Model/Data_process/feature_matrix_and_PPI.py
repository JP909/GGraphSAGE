import argparse

import pandas as pd
import numpy as np
import pyhocon
from tqdm import tqdm
import pandas as pd
def featureMatrix_and_PPInetwork(feature, PPI_network,config):
    print("Generating feature Matrix...")
    idx = list(feature['GeneName'])
    a = list(set(PPI_network.columns).intersection(idx))  ##取feature和PPI_network交集
    adj = PPI_network.loc[a, a]
    feature = feature[feature['GeneName'].isin(a)]
    feature.fillna(0, inplace=True)
    feature.to_csv("D:\\GGraphSAGE\\Data\\LUAD\\LUAD_test.csv")
    np.savetxt(config['file_path']['feature_Matrix'], feature, fmt='%s')
    print(feature)
    print("Generate feature Matrix complete")
    adj.iloc[1, :] = 1
    adj.iloc[:, 1] = 1
    print("building PPI_network")
    source = []
    target = []
    c = 0
    for i1 in tqdm(adj.columns):
        c += 1
        for j in adj.index:
            if adj.loc[i1, j] == 1:
                source.append(i1)
                target.append(j)


    ##定义dataframe
    weight = []
    for i in range(len(source)):
        weight.append(1)
    dict = {'source': source, 'target': target}
    # print(dict)
    data = pd.DataFrame(dict)
    np.savetxt(config['file_path']['PPI_network'], data, fmt='%s')
    print("build PPI network complete")