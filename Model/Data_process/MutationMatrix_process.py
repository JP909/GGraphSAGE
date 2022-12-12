import pandas as pd
from tqdm import tqdm
def create_Mutation_matix(Mutation_dna_basicGroup_raw_data,T_data):
    print("Generating Mutation matrix ...")##获得突变数据
    Gene_list=Mutation_dna_basicGroup_raw_data['gene']##基因对应的患者
    ##生成Mutation
    T_data.columns = T_data.columns.str[:16]
    T_data = T_data.drop(
        T_data[T_data.index.str.startswith('?')].index, axis=0)
    comm = [i for i in T_data.columns if i in Gene_list.index]
    Gene_list = Gene_list[comm]
    Mutations = pd.DataFrame(columns=set(Gene_list.index), index=set(Gene_list.values))
    Mutations.fillna(0, inplace=True)

    for i in tqdm(range(len(Gene_list))):
        Mutations.loc[Gene_list.values[i], Gene_list.index[i]] = 1
    print("Generate Mutation matrix complete")
    print(Mutations)
    return Mutations
