import sys
import os
import torch
import argparse
import pyhocon
import random

from dataCenter import *
from utils import *
from models import *
from Model.Data_process.data_generate import data_generate_func
parser = argparse.ArgumentParser(description='pytorch version of GGraphSAGE')

parser.add_argument('--dataSet', type=str, default='cora')
parser.add_argument('--agg_func', type=str, default='MEAN')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--b_sz', type=int, default=50)
parser.add_argument('--seed', type=int, default=800)
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
parser.add_argument('--gcn', action='store_true')
parser.add_argument('--learn_method', type=str, default='sup')
parser.add_argument('--unsup_loss', type=str, default='normal')
parser.add_argument('--max_vali_f1', type=float, default=0)
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--config', type=str, default='C:\\Users\\MR\Desktop\\GGraphSAGE\\Model\\GGraphSAGE-pytorch-master\\src\\experiments.conf')
args = parser.parse_args()

if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	else:
		device_id = torch.cuda.current_device()
		print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
print('DEVICE:', device)

if __name__ == '__main__':
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	# load config file
	config = pyhocon.ConfigFactory.parse_file(args.config)

	# load data
	# if you prepare all the data we need, you can open the following line to generate your own feature_matrix
	# if you want to run the example, you can strictly run the main method.
	# data_generate_func(config)
	ds = args.dataSet
	dataCenter = DataCenter(config)
	dataCenter.load_dataSet(ds)
	features = torch.FloatTensor(getattr(dataCenter, ds+'_feats')).to(device)#特征矩阵
	idx_test=torch.FloatTensor(getattr(dataCenter,ds+'_test')).to(device)

	graphSage = GraphSage(config['setting.num_layers'], features.size(1), 2, features, getattr(dataCenter, ds+'_adj_lists'), device, gcn=args.gcn, agg_func=args.agg_func)
	graphSage.to(device)

	num_labels = len(set(getattr(dataCenter, ds+'_labels'))) #label的数量

	unsupervised_loss = UnsupervisedLoss(getattr(dataCenter, ds+'_adj_lists'), getattr(dataCenter, ds+'_train'), device)#无监督的loss
	#		判断是无监督还是有监督
	if args.learn_method == 'sup': ##选择有监督
		print('GraphSage with Supervised Learning')
	elif args.learn_method == 'plus_unsup':
		print('GraphSage with Supervised Learning plus Net Unsupervised Learning')
	else:
		print('GraphSage with Net Unsupervised Learning')
#开始训练


	for epoch in range(args.epochs):
		print('----------------------EPOCH %d-----------------------' % epoch)
		graphSage = apply_model(dataCenter, ds, graphSage, unsupervised_loss, args.b_sz, args.unsup_loss, device, args.learn_method)
		if (epoch+1)% 2 == 0 and args.learn_method == 'unsup':
			classification, args.max_vali_f1 = train_classification(dataCenter, graphSage, classification, ds, device, args.max_vali_f1, args.name)
		if args.learn_method != 'unsup':
			args.max_vali_f1,Ft = evaluate(dataCenter, ds, graphSage, device, args.max_vali_f1, args.name, epoch)
