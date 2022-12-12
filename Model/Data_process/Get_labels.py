from tqdm import tqdm
def getLabels(pos_label_data, feature):
    print("Generating labels...")
    labels = []
    for gi in tqdm(feature['GeneName']):
        if gi not in pos_label_data.index:
            labels.append(0)
        else:
            labels.append(1)
    feature['labels']=labels
    print(feature)
    print("Generate labels complete.")
    return feature