from operator import itemgetter
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
def NS_C(Outlier_Matrix, Mutation_Matrix, influenceGragh,feature):
    print("Generating NSC...")
    # commGenes = [i for i in Outlier_Matrix.index if i in Mutation_Matrix.index]
    # Outlier_Matrix = Outlier_Matrix.loc[commGenes, :].T.astype(int)
    # Mutation_Matrix = Mutation_Matrix.loc[commGenes, :].T.astype(int)
    std = []
    cnt = {}
    sox = set(Outlier_Matrix.index)
    smx = set(Mutation_Matrix.index)
    inter = sox.intersection(smx)  ##取交集
    outlier = Outlier_Matrix.loc[inter, :]  ##删除多余的患者
    mutation = Mutation_Matrix.loc[inter, :]
    p = mutation.index
    tmp = {}
    for gi in tqdm(mutation.columns.values):  # 满足gi在MutMartix


        paitient = set()
        cnt[gi] = 0
        for gj in outlier.columns.values:  # 满足gj在outlierMartix
            if gi == gj: continue
            if (gi) in influenceGragh:

                if (gj) in influenceGragh:
                    if influenceGragh.loc[gi, gj] == 1:  # 满足存在已知共表达

                        patients = np.argwhere(np.array(mutation[gi]).astype(int) & np.array(outlier[gj]).astype(
                            int))  ##gi的一行找出来和gj的一行找出来之后乘起来，返回样本索引

                        for pa in patients:

                            if not gi in tmp.keys():
                                tmp[gi] = []

                            tmp[gi].append([gj, p[pa[0]]])  ##基因j和基因j对应的患者
                            paitient.add(p[pa[0]])
        if gi in tmp.keys():
            temp = tmp[gi]

            D1 = map(itemgetter(1), temp)
            d2 = list(D1)

            d2 = set(d2)

            std.append([gi, len(d2)])
            
        else:
            std.append([gi, 0])
            

    std = dict(std)
    comm = [i for i in feature['GeneName'] if i in std.keys()]
    feature = feature[feature["GeneName"].isin(comm)]
    for i in comm:
        feature.loc[(feature['GeneName'] == i), "NSC"] = std[i]

    feature["NSC"] = preprocessing.scale(feature["NSC"].values)
    print(feature)

    print("Generate bipartitle_LCGRG_dict complete")



    print("Generate NSC complete")
    return feature