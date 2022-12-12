import numpy as np
from sklearn import preprocessing


def Methylation(Methylation,feature):
    print("Generating methylation feature...")
    Gd = list(feature['GeneName'])
    Gc = list(Methylation['gene'])
    comm = [val for val in Gd if val in Gc]
    Methylation = Methylation[Methylation['gene'].isin(comm)]
    Methylation = Methylation.groupby('gene').mean()  ##甲基化取重复探针的平均值
    M_normal = Methylation[Methylation.columns[np.where(Methylation.columns.str[-3] == '1')]]
    M_tumor = Methylation[Methylation.columns[np.where(Methylation.columns.str[-3] == '0')]]
    if M_normal.empty==False:
        Methylation_normal_mean = np.mean(M_normal, axis=1)
        Methylation_tumor_mean=np.mean(M_tumor,axis=1)
        Methylation=Methylation_tumor_mean/Methylation_normal_mean
    if M_normal.empty==True:
        Methylation=np.mean(Methylation,axis=1)
    Methylation.fillna(0, inplace=True)
    Methylation = preprocessing.scale(Methylation.values)
    feature = feature[feature['GeneName'].isin(comm)]

    feature['Methylation'] = Methylation
    print(feature)
    print("Generate methylation feature complete.")
    return feature