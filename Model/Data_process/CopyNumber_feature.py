from sklearn import preprocessing
import numpy as np
def CopyNumber_feature(Copy_Numbers,feature):
    print("Generating copy number feature...")
    Gd = list(feature['GeneName'])
    Gc = list(Copy_Numbers.index)
    comm = [val for val in Gd if val in Gc]
    Copy_Numbers = Copy_Numbers.loc[comm, :]
    copySum = np.sum(Copy_Numbers, axis=1)
    Copy_Numbers = preprocessing.scale(copySum.values)
    feature = feature[feature['GeneName'].isin(comm)]
    feature['CopyNumber'] = Copy_Numbers
    print(feature)
    print("Generate copy number feature complete.")
    return feature
