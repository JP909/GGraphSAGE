# GGraphSAGE
We first generate comprehensive instructive features for each gene from genomic, epigenomic,
transcriptomic level together with protein–protein interaction (PPI) networks derived characteristics,
and then propose a novel semi-supervised deep graph learning framework GGraphSAGE to predict cancer
driver genes according to the impact of the alterations on biological system.

If you wish to run our example data, you need to download and generate the gene-patient copy number
and methylation fold change matrix on TCGA, as well as download the PPI network on String and
preserve only the edges with a correlation greater than 990.
We've provided methods for generating each feature, and if you want to run your own data on GGraphSAGE,
You can use "from Model.Data_process.data_generate import data_generate_func" to generate feature matrices
and networks after each type of omics data is collected. Then run the "main" method in" GGmain" to train GGraphSAGE.
