import numpy as np
from sklearn import preprocessing
def Fold_change(N_data,T_data,feature):
    #导入癌症样本和正常样本
    print("Generating fold change feature...")
    normal_mean=N_data.mean(axis=1)
    tumor_mean=T_data.mean(axis=1)
    fold=(tumor_mean)/(normal_mean)
    fold=fold.replace([np.inf, -np.inf], np.nan)
    fold=fold.fillna(0)

    comm=[i for i in feature['GeneName'] if i in fold.index]
    feature = feature[feature['GeneName'].isin(comm)]
    fold=fold[comm]
    feature["fold_change"]=preprocessing.scale(fold.values)
    print(feature)
    print("Generate fold change_feature complete")

    return feature