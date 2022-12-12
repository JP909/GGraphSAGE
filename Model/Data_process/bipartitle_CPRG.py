from tqdm import tqdm
import pandas as pd
from sklearn import preprocessing
def bip_CPRG(CPRGs, Mutation_Matrix):
    print("Generating bipartitle_CPRG_dict...")
    CPRG ={}
    # Mutation_Matrix = Mutation_Matrix.T.astype(int)
    for i in tqdm(Mutation_Matrix.columns):
        if i in CPRGs:
            CPRG[i] = 1
        else:
            CPRG[i] = 0
    print("Generate bipartitle_CGRG_dict complete")
    feature=pd.DataFrame(columns=["GeneName", "bipartitle_CPRG"],index=range(len(CPRG.keys())))
    feature['GeneName']=list(CPRG.keys())
    feature["bipartitle_CPRG"]=list(CPRG.values())
    feature["bipartitle_CPRG"]=preprocessing.scale(feature['bipartitle_CPRG'].values)
    print(feature)

    return feature
