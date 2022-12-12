import pandas as pd
from tqdm import tqdm
# from Model.Data_process.bipartitle_features import bipartitle_main
from sklearn import preprocessing
# feature = bipartitle_main.bipartitle_start()
#
# Mutation_dna_basicGroup_raw_data = pd.read_csv("D:/GGraphSAGE/Data/BRCA/TCGA-BRCA.muse_snv.tsv", sep="\t",
#                                                    header=0, index_col=0)
def base_sub_features(Mutation_dna_basicGroup_raw_data, feature):
    print("Generating base_sub_features...")
    Mutation_dna_basicGroup_raw_data["New"] = Mutation_dna_basicGroup_raw_data["ref"] + \
                                              Mutation_dna_basicGroup_raw_data["alt"]
    comm = [val for val in list(feature["GeneName"]) if val in list(Mutation_dna_basicGroup_raw_data["gene"])]
    Mutation_dna_basicGroup_raw_data = Mutation_dna_basicGroup_raw_data[
        Mutation_dna_basicGroup_raw_data["gene"].isin(comm)]
    Mutation_dna_basicGroup_raw_data["NEW2"] = Mutation_dna_basicGroup_raw_data["gene"] + "+" + \
                                               Mutation_dna_basicGroup_raw_data["New"]
    tm = Mutation_dna_basicGroup_raw_data.groupby("NEW2").count()
    tm['num'] = tm['gene']
    tm['idx'] = tm.index
    tm = tm.loc[:, ['num', 'idx']]
    tm['idx'].str.split('+', expand=True)
    tm['gene'] = tm['idx'].str.split('+', expand=True)[0]
    tm['alt'] = tm['idx'].str.split('+', expand=True)[1]
    del (tm['idx'])
    base_features = pd.DataFrame(columns=set(tm['alt']), index=set(tm['gene']))
    for i in tqdm(range(len(tm.index))):
        base_features.loc[tm.iloc[i, 1], tm.iloc[i, 2]] = tm.iloc[i, 0]
    base_features.fillna(0, inplace=True)
    base_features = base_features.loc[comm, :]
    print("Data are being standardized...")
    for i in tqdm(base_features.columns):
        base_features.loc[:, i] = preprocessing.scale(base_features.loc[:, i].values)
    feature = feature[feature['GeneName'].isin(comm)]
    feature[list(base_features)] = base_features.values
    col = []
    for i in feature.columns:
        if i.startswith('-'):
            col.append("'" + i)
        else:
            col.append(i)
    feature.columns = col
    print(feature)
    print("Generate base_sub_features complete")
    return feature
# base_sub_features(Mutation_dna_basicGroup_raw_data,feature)