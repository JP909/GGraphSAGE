from tqdm import tqdm
from sklearn import preprocessing
def bipartitle_L(Outlier_Matrix, Mutation_Matrix, influenceGragh,cell_proliferation_related_genes,feature):
    print("Generating bipartitle_LCPRG_dict...")
    L_edges = {}
    # commGenes = [i for i in Outlier_Matrix.index if i in Mutation_Matrix.index]
    # Outlier_Matrix = Outlier_Matrix.loc[commGenes, :].T.astype(int)
    # Mutation_Matrix = Mutation_Matrix.loc[commGenes, :].T.astype(int)
    sox = set(Outlier_Matrix.index)
    smx = set(Mutation_Matrix.index)
    inter = sox.intersection(smx)  ##取交集
    outlier = Outlier_Matrix.loc[inter, :]  ##删除多余的患者

    mutation = Mutation_Matrix.loc[inter, :]

    for gi in tqdm(mutation.columns.values):  # 满足gi在MutMartix
        L_edges[gi] = 0
        for gj in outlier.columns.values:  # 满足gj在outlierMartix
            if gi == gj: continue
            if (gi) in influenceGragh:
                if (gj) in influenceGragh:
                    if influenceGragh.loc[gi, gj] == 1:
                        if gj in cell_proliferation_related_genes:
                            L_edges[gi] += 1
    comm = [i for i in feature['GeneName'] if i in L_edges.keys()]
    feature = feature[feature["GeneName"].isin(comm)]
    for i in comm:
        feature.loc[(feature['GeneName'] == i), "bipartitle_LCPRG"] = L_edges[i]

    feature["bipartitle_LCPRG"] = preprocessing.scale(feature["bipartitle_LCPRG"].values)
    print(feature)

    print("Generate bipartitle_LCPRG_dict complete")

    return feature
