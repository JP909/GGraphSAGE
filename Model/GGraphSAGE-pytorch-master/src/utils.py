import sys
import os

import pyhocon
import torch
import random
import math

from sklearn.metrics import *
#from matplotlib import pyplot as plt
from sklearn.utils import shuffle
#from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch.nn as nn
import argparse
import numpy as np
import pandas as pd
parser = argparse.ArgumentParser(description='pytorch version of GGraphSAGE')
parser.add_argument('--config', type=str, default='D:\\GGraphSAGE\Model\\GGraphSAGE-pytorch-master\\src\\experiments.conf')
args = parser.parse_args()
config = pyhocon.ConfigFactory.parse_file(args.config)
b=pd.read_csv(config["file_path"]["newFind"], header=0, index_col=0)
label_discor=b.values
def evaluate(dataCenter, ds, graphSage, device, max_vali_f1, name, cur_epoch):
	test_nodes = getattr(dataCenter, ds+'_test')
	val_nodes = getattr(dataCenter, ds+'_val')
	labels = getattr(dataCenter, ds+'_labels')
	dict1=getattr(dataCenter,ds+'_dict')
	assert (isinstance(dict1,dict))
	##验证集的基因

	model = graphSage

	params = []

	for param in model.parameters():
		if param.requires_grad:
			param.requires_grad = False
			params.append(param)

	embs = graphSage(val_nodes)
	logists = embs

	_, predicts = torch.max(logists, 1)
	labels_val = labels[val_nodes]##验证集真实值
	assert len(labels_val) == len(predicts)
	print("验证集真实值",labels_val)
	print("预测值",predicts)
	c=0
	d=0
	cor=[]
	discor=[]
	for i in range(len(predicts)):
		if predicts.numpy().tolist()[i]==1:
			if labels_val[i] == 1:
				c+=1
				cor.append(val_nodes[i])
			else:
				d+=1
				discor.append(val_nodes[i])
	correct_gene = [k for k, v in dict1.items() if v in cor]
	discorrect_gene=[k for k,v in dict1.items() if v in discor]
	TP=c
	FP=d
	print("验证集新发现",FP)
	print("验证集真阳性",TP)
	print("验证集一共有",sum(labels_val))
	print("验证集预测正确",c/sum(labels_val))
	print("验证集预测正确的基因",correct_gene)
	print("验证集预测错误的基因",discorrect_gene)
# print("PPV",TP/(FP+TP))


	vali_f1 = f1_score(labels_val, predicts.cpu().data, average="micro")
	print("Validation F1:", vali_f1)

	embs_test = graphSage(test_nodes)
	logists_test = embs_test
	_, test_predicts = torch.max(logists_test, 1)
	labels_test = np.array(labels)[test_nodes]
	assert len(labels_test) == len(test_predicts)

	AP = average_precision_score(labels_test, logists_test[:,1], average='macro', pos_label=1, sample_weight=None)
	print("AP",AP)


	print("测试集真实值", labels_test)
	print("测试集预测值", test_predicts)
	cor2 = 0
	for i in range(len(labels_test)):
		if labels_test[i] == test_predicts[i]:
			cor2 += 1
	TP_t = 0
	FP_t = 0
	FN_t = 0
	cor = []
	discor = []
	zhen_t = []
	for i in range(len(labels_test)):
		if labels_test[i] == 1:
			zhen_t.append(test_nodes[i])
	print("真值T为", [k for k, v in dict1.items() if v in zhen_t])
	for i in range(len(test_predicts)):
		if test_predicts.numpy().tolist()[i] == 1:
			if labels_test[i] == 1:
				TP_t += 1
				cor.append(test_nodes[i])
			else:
				FP_t += 1
				discor.append(test_nodes[i])
		if test_predicts.numpy().tolist()[i] == 0:
			if labels_test[i] == 1:
				FN_t += 1
	correct_gene = [k for k, v in dict1.items() if v in cor]
	discorrect_gene = [k for k, v in dict1.items() if v in discor]
	New_finder = []
	if FP_t == 0:
		FP_t = 0.001
	if FN_t == 0:
		FN_t = 0.001
	P_t = TP_t / (TP_t + FP_t)
	R_t = TP_t / (TP_t + FN_t)
	print("测试集新发现", FP_t)
	print("测试集真阳性", TP_t)
	print("测试集一共有", sum(labels_test))
	print("测试集预测正确", TP_t / sum(labels_test))
	print("Precision_t", P_t)
	print("Recall_t", R_t)



	print("测试集预测正确的基因", correct_gene)
	print("测试集预测错误的基因", discorrect_gene)

	# print("PPV", TP / (FP + TP))

	test_f1 = f1_score(labels_test, test_predicts.cpu().data, average="micro")
	print("Test F1:", test_f1)

	for param in params:
		param.requires_grad = True

	# torch.save(models, 'models/model_best_{}_ep{}_{:.4f}.torch'.format(name, cur_epoch, test_f1))torch.save(models, 'models/model_best_{}_ep{}_{:.4f}.torch'.format(name, cur_epoch, test_f1))

	for param in params:
		param.requires_grad = True

	return max_vali_f1, FP_t


def get_gnn_embeddings(gnn_model, dataCenter, ds):
    print('Loading embeddings from trained GraphSAGE model.')
    features = np.zeros((len(getattr(dataCenter, ds+'_labels')), gnn_model.out_size))
    nodes = np.arange(len(getattr(dataCenter, ds+'_labels'))).tolist()
    b_sz = 500
    batches = math.ceil(len(nodes) / b_sz)
    embs = []
    for index in range(batches):
        nodes_batch = nodes[index*b_sz:(index+1)*b_sz]
        embs_batch = gnn_model(nodes_batch)
        assert len(embs_batch) == len(nodes_batch)
        embs.append(embs_batch)
        # if ((index+1)*b_sz) % 10000 == 0:
        #     print(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')

    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    assert len(embs) == len(nodes)
    print('Embeddings loaded.')
    return embs.detach()

def train_classification(dataCenter, graphSage, ds, device, max_vali_f1, name, epochs=800):
	print('Training Classification ...')
	c_optimizer = torch.optim.Adam(graphSage.parameters(),lr =0.05, weight_decay=0.0005)
	# train classification, detached from the current graph
	#classification.init_params()
	b_sz = 50
	train_nodes = getattr(dataCenter, ds+'_train')
	labels = getattr(dataCenter, ds+'_labels')
	features = get_gnn_embeddings(graphSage, dataCenter, ds)
	for epoch in range(epochs):
		train_nodes = shuffle(train_nodes)
		batches = math.ceil(len(train_nodes) / b_sz)
		visited_nodes = set()
		for index in range(batches):
			nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]
			visited_nodes |= set(nodes_batch)
			labels_batch = labels[nodes_batch]
			embs_batch = features[nodes_batch]

			logists = graphSage(embs_batch)
			loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
			loss /= len(nodes_batch)
			# print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))

			loss.backward()

			nn.utils.clip_grad_norm_(graphSage.parameters(), 5)
			c_optimizer.step()
			c_optimizer.zero_grad()

		max_vali_f1 = evaluate(dataCenter, ds, graphSage, device, max_vali_f1, name, epoch)
	return  max_vali_f1

def apply_model(dataCenter, ds, graphSage, unsupervised_loss, b_sz, unsup_loss, device, learn_method):
	test_nodes = getattr(dataCenter, ds+'_test')
	val_nodes = getattr(dataCenter, ds+'_val')
	train_nodes = getattr(dataCenter, ds+'_train')
	labels = getattr(dataCenter, ds+'_labels')

	if unsup_loss == 'margin':
		num_neg = 6
	elif unsup_loss == 'normal': ##有监督学习的loss
		num_neg = 100
	else:
		print("unsup_loss can be only 'margin' or 'normal'.")
		sys.exit(1)

	train_nodes = shuffle(train_nodes) ##把训练集打乱

	model = graphSage ##有两个模型组成
	params = []
	#存参数

	for param in model.parameters():
		if param.requires_grad:
			params.append(param)
	#优化器，梯度下降
	optimizer = torch.optim.SGD(params, lr=0.01)
	optimizer.zero_grad()

	model.zero_grad()

	batches = math.ceil(len(train_nodes) / b_sz) ##batchsize为20

	count=0
	#对每个batch训练
	for index in range(batches):
		nodes_batch = train_nodes[(index*b_sz):((index+1)*b_sz)]

		count += len(nodes_batch)
		# extend nodes batch for unspervised learning
		# no conflicts with supervised learning
		nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=50)))


		# get ground-truth for the nodes batch
		labels_batch = labels[nodes_batch]

		# feed nodes batch to the graphSAGE
		# returning the nodes embeddings
		embs_batch = graphSage(nodes_batch)

		if learn_method == 'sup':
			# superivsed learning
			logists = embs_batch
			loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
			loss_sup /= len(nodes_batch)
			loss = loss_sup
		elif learn_method == 'plus_unsup':
			# superivsed learning
			logists = embs_batch
			loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
			loss_sup /= len(nodes_batch)
			# unsuperivsed learning
			if unsup_loss == 'margin':
				loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
			elif unsup_loss == 'normal':
				loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
			loss = loss_sup + loss_net
		else:
			if unsup_loss == 'margin':
				loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
			elif unsup_loss == 'normal':
				loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
			loss = loss_net

		print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index+1, batches, loss.item(), count, len(train_nodes)))
		loss.backward()

		nn.utils.clip_grad_norm_(model.parameters(), 5)
		optimizer.step()

		optimizer.zero_grad()

		model.zero_grad()

	return graphSage