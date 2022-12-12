
import dgl
import numpy as np
import torch as th
import pandas as pd
from dgl.nn import GATConv
from collections import defaultdict
import numpy as np

class DataCenter(object):
	"""docstring for DataCenter"""
	def __init__(self, config):
		super(DataCenter, self).__init__()
		self.config = config
		
	def load_dataSet(self, dataSet='cora'):
		cora_content_file = self.config['file_path']['feature_Matrix']
		cora_cite_file = self.config['file_path']['PPI_network']
		feat_data = []  ##存feature
		labels = []  # label sequence of node
		node_map = {}  # 将节点编码存入node_map
		label_map = {}  # 将label量化编码存入label_map
		with open(cora_content_file) as fp:
			for i, line in enumerate(fp):
				info = line.strip().split()
				feat_data.append([float(x) for x in info[1:-1]])
				node_map[info[0]] = i
				if not info[-1] in label_map:
					label_map[info[-1]] = len(label_map)
				labels.append(label_map[info[-1]])
		feat_data = np.asarray(feat_data)
		labels = np.asarray(labels, dtype=np.int64)
		print(len(labels))

		b = ([], [])
		with open(cora_cite_file) as fp:
			for i, line in enumerate(fp):
				info = line.strip().split()
				assert len(info) == 2
				paper1 = node_map[info[0]]
				paper2 = node_map[info[1]]
				b[0].append(paper1)
				b[1].append(paper2)  ##将蛋白质关系网络改成字典形式
		g = dgl.graph(b)
		g = dgl.add_self_loop(g)
		gatconv = GATConv(self.config['setting']["GAT_input_layer"], self.config['setting']["GAT_output_layer"],
						  num_heads=self.config['setting']["GAT_num_heads"])
		feat = feat_data
		feat = th.tensor(feat, dtype=th.float)
		res = gatconv(g, feat)
		res = res.mean(keepdim=False, axis=1)

		feat_data = res.detach().numpy()

		adj_lists = defaultdict(set)
		with open(cora_cite_file) as fp:
			for i,line in enumerate(fp):
				info = line.strip().split()
				assert len(info) == 2
				paper1 = node_map[info[0]]
				paper2 = node_map[info[1]]
				adj_lists[paper1].add(paper2)
				adj_lists[paper2].add(paper1) ##将蛋白质关系网络改成字典形式

		assert len(feat_data) == len(labels)==len(adj_lists)
			##分割数据集
		test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

		setattr(self, dataSet+'_test', test_indexs)
		setattr(self, dataSet+'_val', val_indexs)
		setattr(self, dataSet+'_train', train_indexs)
		setattr(self,dataSet+ '_adj',fp)
		setattr(self, dataSet+'_feats', feat_data)
		setattr(self, dataSet+'_labels', labels)
		setattr(self, dataSet+'_adj_lists', adj_lists)
		setattr(self,dataSet+'_dict',node_map)


	def _split_data(self, num_nodes, test_split = 3, val_split = 3):
		rand_indices = np.random.permutation(num_nodes)

		test_size = num_nodes // test_split
		val_size = num_nodes // val_split
		train_size = num_nodes - (test_size + val_size)

		test_indexs = rand_indices[1400:]
		val_indexs = rand_indices[100:1200]
		train_indexs = rand_indices[:1400]
		
		return test_indexs, val_indexs, train_indexs


