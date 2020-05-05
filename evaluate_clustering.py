import pickle
import csv
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import seaborn as sns
import pandas as pd

def make_cluster_dict(tree_path, index_path):
	index_dict = pickle.load(open(index_path, 'rb'))
	cluster_dict = dict()
	max_depth = 0
	depth_dict = dict()

	with open(tree_path, 'r') as f:
		for line in f:
			clusters = line.split(' ')
			clusters[-1] = clusters[-1][:-1]
			if len(clusters) - 2 > max_depth:
				max_depth = len(clusters) - 2
			for x in range(2, len( clusters)):
				depth_dict[int(clusters[x])] = x - 2
				if int(clusters[x]) not in cluster_dict:
					if x == 2:
						cluster_dict[int(clusters[x])] = [(-1, 0), [index_dict[int(clusters[0])]]]
					else:
						cluster_dict[int(clusters[x])] = [(int(clusters[x-1]), x-2), [index_dict[int(clusters[0])]]]

				else:
					temp = cluster_dict[int(clusters[x])]
					temp[1].append(index_dict[int(clusters[0])])
					cluster_dict[int(clusters[x])] = temp

	
	return cluster_dict,depth_dict, max_depth

def make_level_box_plot(cluster_dict_1, depth_dict_1, max_depth_1, cluster_dict_2, depth_dict_2, max_depth_2, method_1, method_2, title, path):
	levels_size = pd.DataFrame(columns = ['Depth', 'Size', 'Method'])
	
	for key in cluster_dict_1:
		cl = cluster_dict_1[key]
		levels_size  = levels_size.append({'Depth': cl[0][1], 'Size': len(cl[1]), 'Method': method_1}, ignore_index=True)

	for key in cluster_dict_2:
		cl = cluster_dict_2[key]
		levels_size  = levels_size.append({'Depth': cl[0][1], 'Size': len(cl[1]), 'Method': method_2}, ignore_index=True)

	ax = sns.boxplot(x="Depth", y="Size", hue="Method",
                 data=levels_size, palette="Set3")
	ax.set_title(title)
	plt.savefig(path)
	plt.show()

def make_level_topic_accuracy(cluster_dict, max_depth, title, path):
	category_set = set()

	for key in cluster_dict:
		for file in cluster_dict[key][1]:
			category = int(file.split('_')[0])
			if category not in category_set:
				category_set.add(category)


	level_score_dict = dict()
	for x in range(0, max_depth):
		level = dict()
		for item in category_set:
			level[item] = [0, 0]
		level_score_dict[x] = level

	for key in cluster_dict:
		cl = cluster_dict[key]
		underlying_dict = dict()
		for cat in category_set:
			underlying_dict[cat] = 0
		max_count = [0, None]
		for file in cluster_dict[key][1]:
			category = int(file.split('_')[0])
			underlying_dict[category] += 1
			if underlying_dict[category] > max_count[0]:
				max_count = [underlying_dict[category], category]

	
		for cat in underlying_dict:
			if cat == max_count[1] and max_count[0]*2>len(cluster_dict[key][1]):
				print(cat)
				temp = level_score_dict[cluster_dict[key][0][1]][cat]
				temp[0] += underlying_dict[cat]
				temp[1] += underlying_dict[cat]
				level_score_dict[cluster_dict[key][0][1]][cat] = temp
			else:
				temp = level_score_dict[cluster_dict[key][0][1]][cat]
				temp[1] += underlying_dict[cat]
				level_score_dict[cluster_dict[key][0][1]][cat] = temp

	q = np.zeros((max_depth, len(category_set)))

	for i in range(0, max_depth):
		lv = level_score_dict[i]
		j = 0
		for cat in category_set:

			q[i][j] = lv[cat][0]/lv[cat][1]
			j += 1
	
	labels = []
	for category in category_set:
		labels.append(category)

	label_dict = {2: 'wall', 6:'flower', 9:'coast', 10:'airplane', 17:'chicken'}

	for i in range(0, len(labels)):
		labels[i] = label_dict[labels[i]]


	ax = sns.heatmap(q,  linewidth=0.5, cbar_kws={'label': 'Grouping Accuracy'})
	ax.set(xlabel='Picture Category', ylabel='Tree Level')
	ax.set_xticklabels( labels)
	ax.set_title(title)
	plt.savefig(path)
	plt.show()




def evaluate_clustering(cluster_dict, max_depth):
	level_score_dict = dict()
	for x in range(0, max_depth):
		level_score_dict[x] = [0, 0]

	for key in cluster_dict:
		cl = cluster_dict[key]
		underlying_dict = dict()
		max_count = [0, None]
		for file in cluster_dict[key][1]:
			category = int(file.split('_')[0])
			if category in underlying_dict:
				underlying_dict[category] += 1
			else:
				underlying_dict[category] = 1

			if underlying_dict[category] > max_count[0]:
				max_count = [underlying_dict[category], category]

		if max_count[0] >= 2:
			temp = level_score_dict[cluster_dict[key][0][1]]
			temp[0] += max_count[0]
			temp[1] += len(cluster_dict[key][1])
			level_score_dict[cluster_dict[key][0][1]] = temp

	return level_score_dict[max_depth-1]
		




if __name__ == '__main__':
	method_1 = 'ORB'
	method_2 = 'XTRACT'
	cluster_dict_1, depth_dict_1, max_depth_1 = make_cluster_dict('n_cuts_runs/ORB_50/mode.assign', 'n_cuts_vecs/order.pkl')
	cluster_dict_2, depth_dict_2, max_depth_2 = make_cluster_dict('n_cuts_runs/XTRACT_50/mode.assign', 'n_cuts_vecs/order.pkl')
	make_level_box_plot(cluster_dict_1, depth_dict_1, max_depth_1, cluster_dict_2, depth_dict_2, max_depth_2, method_1, method_2, 'ORB vs XTRACT Cluster Density', 'density.jpg')
	#make_level_hist(cluster_dict, depth_dict, max_depth)
	make_level_topic_accuracy(cluster_dict_1, max_depth_1, 'ORB by Topic and Level Grouping Accuracy', 'orb.jpg')
	make_level_topic_accuracy(cluster_dict_2, max_depth_2, 'XTRACT by Topic and Level Grouping Accuracy', 'xtract.jpg')

	'''
	with open('results.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerow(['Segmentation Technique', 'Feature Extraction Technique', 'Number of Words in Vocab', 'Level', 'Accuracy'])
		for folder in ['n_cuts', 'pytorch_cnn', 'k_means']:
			for vec_tech in ['HIST', 'ORB', 'XTRACT']:
				for num in ['20', '30', '50', '100']:
					scores = extract_score('{}_runs/{}_{}/mode.assign'.format(folder, vec_tech, num), '{}_vecs/order.pkl'.format(folder))
					for i in range(0, len(scores)):
						accuracy = scores[i][0]/scores[i][1]
						writer.writerow([folder, vec_tech, num, i+1, accuracy])
	'''
	