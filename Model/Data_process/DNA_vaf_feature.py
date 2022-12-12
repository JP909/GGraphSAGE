def DNA_vaf(Mutation_dna_basicGroup_raw_data,feature):
    print("Generating DNA_vaf feature...")
    c = Mutation_dna_basicGroup_raw_data.groupby(['gene']).mean()
    c_gene = c.index
    geneName = feature["GeneName"]
    comm = [val for val in geneName if val in c_gene]
    c = c.loc[comm, :]
    feature = feature[feature["GeneName"].isin(comm)]
    feature["DNA_vaf"] = c["dna_vaf"].values
    print("Generate DNA_vaf feature complete.")
    print(feature)
    return feature