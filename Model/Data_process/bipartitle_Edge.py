from tqdm import tqdm
from sklearn import preprocessing
def bipartitle_E(Outlier_Matrix,Mutation_Matrix, PPInetwork,feature):
    print("Generating bipartitle_Edge...")
    commGenes = [i for i in Outlier_Matrix.index if i in Mutation_Matrix.index]
    # Outlier_Matrix = Outlier_Matrix.loc[commGenes, :].T.astype(int)
    # Mutation_Matrix = Mutation_Matrix.loc[commGenes, :].T.astype(int)
    cou4 = 0
    cnt = {}
    sox = set(Outlier_Matrix.index)
    smx = set(Mutation_Matrix.index)
    inter = sox.intersection(smx)  ##取交集
    outlier = Outlier_Matrix.loc[inter, :]  ##删除多余的患者

    mutation = Mutation_Matrix.loc[inter, :]

    for gi in tqdm(mutation.columns.values):  # 满足gi在MutMartix

        cou4 += 1

        cnt[gi] = 0

        for gj in outlier.columns.values:  # 满足gj在outlierMartix
            if gi == gj: continue
            if (gi) in PPInetwork:
                if (gj) in PPInetwork:
                    if PPInetwork.loc[gi, gj] == 1:
                        cnt[gi] += sum(mutation[gi] & outlier[gj])
    print("Generate bipartitle_Edge complete")
    comm=[i for i in feature['GeneName'] if i in cnt.keys()]
    feature = feature[feature["GeneName"].isin(comm)]
    for i in comm:
        feature.loc[(feature['GeneName']==i),"bipartitle_E"]=cnt[i]

    feature["bipartitle_E"]=preprocessing.scale(feature["bipartitle_E"].values)
    print(feature)
    return feature
