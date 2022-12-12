import pandas as pd
import numpy as np
import scipy.stats as stats
from tqdm import tqdm

def Transform(Tumor_Gene_expression_data, alpha):
    print("Generating Outlier matrix...")
    ##将基因表达矩阵的马氏距离计算出来并且计算|mean-xi|/s
    def GrubbsTests(Tumor_Gene_expression_data):
        mean_list = []
        mean_list2 = []
        for i in Tumor_Gene_expression_data.columns:  # 对基因表达矩阵进行遍历

            col_mean = np.mean(Tumor_Gene_expression_data.loc[:, i])  # 对expData取列求均值

            mean_list.append(col_mean)

        mean_array = np.array(mean_list)  # 转换数据类型
        dt = Tumor_Gene_expression_data.to_numpy()  # 转换成numpy数据类型
        distance = np.sqrt(abs(dt - mean_array)) * 100  ##求距离矩阵
        distance = pd.DataFrame(distance)
        distance.columns = Tumor_Gene_expression_data.columns
        for i in distance.columns:
            col_mean = np.mean(distance.loc[:, i])
            mean_list2.append(col_mean)
        mean_array2 = np.array(mean_list2)
        distanceArray = distance.to_numpy()
        cal_array = abs(distanceArray - mean_array2)  ##求出|Xi-mean|
        std = np.array(np.std(distance, axis=0)).reshape((1, len(Tumor_Gene_expression_data.columns)))
        for i in range(std.shape[1]):
            if any(std[:, i] == 0):
                std[:, i] = 0.0001

        Gcalculated = cal_array / std
        G1 = pd.DataFrame(Gcalculated)
        G1.index = Tumor_Gene_expression_data.index


        return G1.to_numpy()

    ##根据采样次数计算上界
    def calculate_critical_value(size, alpha):
        t_dist = stats.t.ppf(1 - alpha / (2 * size), size - 2)
        numerator = (size - 1) * np.sqrt(np.square(t_dist))
        denominator = np.sqrt(size) * np.sqrt(size - 2 + np.square(t_dist))
        critical_value = numerator / denominator
        print("Grubbs Critical Value:{}".format(critical_value))
        return critical_value

    G = calculate_critical_value(len(Tumor_Gene_expression_data.index), alpha)
    tmp = GrubbsTests(Tumor_Gene_expression_data)
    for i in tqdm(range(tmp.shape[0])):
        for j in range(tmp.shape[1]):
            if G < tmp[i][j]:
                tmp[i][j] = 1
            else:
                tmp[i][j] = 0
    Outlier = pd.DataFrame(tmp, columns=Tumor_Gene_expression_data.columns, index=Tumor_Gene_expression_data.index)

    print("Generate Outlier matrix complete")
    print(Outlier)

    return Outlier



