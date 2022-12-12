
def Mut_Bio_features(feature,Mut_bio_file):
    print("Generating MutSigCV and BioGrid feature...")
    comm=[i for i in feature["GeneName"] if i in Mut_bio_file.index]
    feature = feature[feature['GeneName'].isin(comm)]
    Mut_bio_file=Mut_bio_file.loc[comm,:]
    # for i in Mut_bio_file.columns:
    #     Mut_bio_file.loc[:, i]=preprocessing.scale(Mut_bio_file.loc[:, i].values)
    feature[list(Mut_bio_file.columns)] = Mut_bio_file.values
    print(feature)
    print("Generate MutSigCV and BioGrid feature complete")
    return feature
