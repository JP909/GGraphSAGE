import numpy as np
from sklearn import preprocessing
from tqdm import tqdm
def Mutation_level(Mutation_Matrix,feature):
    print("Generating Mutation level feature...")
    sumM = np.sum(Mutation_Matrix, axis=0)
    Mutation = []
    sumM = dict(sumM)
    for i in tqdm(feature['GeneName']):
        Mutation.append(sumM[i])
    feature['Mutation_level'] = preprocessing.scale(Mutation)
    print(feature)
    print("Generate Mutation level feature complete.")
    return feature