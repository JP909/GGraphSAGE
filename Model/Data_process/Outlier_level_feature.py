import numpy as np
from sklearn import preprocessing
from tqdm import tqdm
def Outlier_level(OutlierMatirx,feature):
    print("Generating outlier level feature...")
    sumO = np.sum(OutlierMatirx, axis=0)

    sumO = dict(sumO)
    Outliers = []
    for i in tqdm(feature['GeneName']):
        Outliers.append(sumO[i])

    feature['Outlier_level'] = preprocessing.scale(Outliers)
    print(feature)
    print("Generate outlier level feature complete.")
    return feature
