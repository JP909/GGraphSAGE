import argparse
import pyhocon
from Model.Data_process import CopyNumber_feature
from Model.Data_process import bipartitle_CPRG
from Model.Data_process import bipartitle_LCPRG
from Model.Data_process import Fold_change_feature
import pandas as pd
from Model.Data_process import DNA_vaf_feature
from Model.Data_process import P_value_feature
from Model.Data_process import Base_substitution_features
from Model.Data_process import MutSigCV_BioGrid_features
from Model.Data_process import Methylation_feature
from Model.Data_process import Outlier_level_feature
from Model.Data_process import Mutation_level_feature
from Model.Data_process import Get_labels
from Model.Data_process import PPI_network_process
from Model.Data_process import MutationMatrix_process
from Model.Data_process import OutlierMatrix_process
from Model.Data_process import feature_matrix_and_PPI
from Model.Data_process import NSC
from Model.Data_process import bipartitle_Edge


def data_generate_func(config):
    protein_links = pd.read_csv(config['file_path']['protein_links'], sep=" ", header=0)  ##蛋白质链接(统一)
    protein_info = pd.read_csv(config['file_path']['protein_info'], sep='\t', header=0)  ##蛋白质网络(统一)
    Mutation_dna_basicGroup_raw_data = pd.read_csv(config['file_path']['Mutation_dna_basicGroup_raw_data'], sep="\t",
                                                   header=0, index_col=0)
    T_data = pd.read_csv(config['file_path']['T_data'], header=0, index_col=0)  ##表达数据index:基因，columns：样本
    N_data = pd.read_csv(config['file_path']['N_data'], header=0, index_col=0)
    Mut_Bio_file = pd.read_csv(config['file_path']['Mut_Bio_file'], header=0, index_col=0)
    Copy_Number = pd.read_csv(config['file_path']['Copy_Number'], header=0, index_col=0)
    Methylation = pd.read_csv(config['file_path']['Methylation'], header=0, index_col=None)
    cell_proliferation_related_genes = pd.read_csv(config['file_path']['Cell_growth_related_genes'], header=0,
                                                   index_col=0).values
    labels = pd.read_csv(config['file_path']['labels'], sep="\t", header=0, index_col=0)
    PPInetwork = PPI_network_process.PPI_create(protein_info, protein_links).astype(int)
    Mutation_Matrix = MutationMatrix_process.create_Mutation_matix(Mutation_dna_basicGroup_raw_data,
                                                                   T_data)

    Outlier_Matrix = OutlierMatrix_process.Transform(Tumor_Gene_expression_data=T_data, alpha=0.5)
    commGenes = [i for i in Outlier_Matrix.index if i in Mutation_Matrix.index]
    Outlier_Matrix = Outlier_Matrix.loc[commGenes, :].T.astype(int)
    Mutation_Matrix = Mutation_Matrix.loc[commGenes, :].T.astype(int)

    feature = bipartitle_CPRG.bip_CPRG(cell_proliferation_related_genes, Mutation_Matrix)
    feature = Methylation_feature.Methylation(Methylation, feature)
    feature = Mutation_level_feature.Mutation_level(Mutation_Matrix, feature)
    feature = Outlier_level_feature.Outlier_level(Outlier_Matrix, feature)
    feature = CopyNumber_feature.CopyNumber_feature(Copy_Number, feature)
    feature = DNA_vaf_feature.DNA_vaf(Mutation_dna_basicGroup_raw_data, feature)
    feature = Fold_change_feature.Fold_change(N_data, T_data, feature)
    feature = P_value_feature.P_values(N_data, T_data, feature)
    feature = Base_substitution_features.base_sub_features(Mutation_dna_basicGroup_raw_data, feature)
    feature = MutSigCV_BioGrid_features.Mut_Bio_features(feature, Mut_Bio_file)
    feature = NSC.NS_C(Outlier_Matrix, Mutation_Matrix, PPInetwork, feature)
    feature = bipartitle_LCPRG.bipartitle_L(Outlier_Matrix, Mutation_Matrix, PPInetwork,
                                            cell_proliferation_related_genes,
                                            feature)
    feature = bipartitle_Edge.bipartitle_E(Outlier_Matrix, Mutation_Matrix, PPInetwork, feature)
    feature = Get_labels.getLabels(labels, feature)
    feature_matrix_and_PPI.featureMatrix_and_PPInetwork(feature, PPInetwork, config)
