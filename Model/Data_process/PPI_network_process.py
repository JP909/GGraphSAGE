import numpy as np
import pandas as pd
from tqdm import tqdm

def PPI_create(protein_info,protein_links):
    # protein_links.columns = ['protein1', 'protein2', 'combined_score']
    print("Generating PPI network...")
    result_dic = dict()
    ##生成名称ID字典
    for i in protein_info.index:
        result_dic[protein_info.loc[i, "preferred_name"]] = protein_info.loc[i, 'protein_external_id']
    protein_990_part = protein_links.iloc[np.where(protein_links.iloc[:, 2] >= 990)[0], :]##查找关系大于990的边
    protein_990_ID = set(protein_990_part.protein1) | set(protein_990_part.protein2)
    influnce = pd.DataFrame(0, index=protein_990_ID, columns=protein_990_ID)
    protein_990_part = protein_990_part.drop(columns="combined_score")


    for i in tqdm(protein_990_part.index):

        ind1 = protein_990_part.loc[i, "protein1"]
        ind2 = protein_990_part.loc[i, "protein2"]
        influnce.loc[ind1, ind2] = 1



    dia_dt = influnce.values
    dia_i = list(range(len(influnce)))
    dia_dt[dia_i, dia_i] = 1
    protien = pd.DataFrame(dia_dt, index=influnce.index, columns=influnce.columns)
    result_dic = dict()
    for i in protein_info.index:
        result_dic[protein_info.loc[i, "preferred_name"]] = protein_info.loc[i, 'protein_external_id']

    result_dic1={}
    for i in protein_info.index:
        result_dic1[protein_info.loc[i, 'protein_external_id']] = protein_info.loc[i, "preferred_name"]
    gene_index = [result_dic1[i] for i in protien.index]
    protien.columns = gene_index
    protien.index = gene_index
    print("Generate PPI network complete")
    print(protien)
    return protien
