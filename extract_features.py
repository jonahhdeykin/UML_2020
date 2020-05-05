import cv2
import numpy as np
import scipy
from skimage.io import imread
import pickle
import random
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle
import sys
from xtract_features.glcms import *
from PIL import Image



def extract_features_KAZE(image_path, vector_size=512):
    image = imread(image_path)
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them. 
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        #kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.detectAndCompute(image, None)
        # Flatten all of them in one big vector - our feature vector
        try:
            dsc = dsc.flatten()
        except:
            dsc = np.zeros(vector_size * 64)
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        print(dsc)
    except cv2.error as e:
        print('Error: ', e)
        return None

    return dsc

def extract_features_ORB(image_path, vector_size=64):
    image = imread(image_path)
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.ORB_create()
        # Dinding image keypoints
        kps = alg.detect(image, None)
        # Getting first 32 of them. 
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        #kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.detectAndCompute(image, None)
        
        # Flatten all of them in one big vector - our feature vector
        try:
            dsc = dsc.flatten()
        except:
            dsc = np.zeros(vector_size * 64)
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        print(dsc)
    except cv2.error as e:
        print('Error: ', e)
        return None

    return dsc
def extract_features_XTRACT(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    feats = glcm(img)
    print(feats)
    return feats.glcm_all()

def build_file(index_dict, clusters, outpath):
    with open(outpath, 'w') as f:
        for picture in index_dict:
            indices = index_dict[picture]
            words = clusters[indices[0]:indices[1]]
            words_dict = dict()
            if len(words) == 0:
                print(picture)
            for item in words:
                if item in words_dict:
                    words_dict[item] += 1
                else:
                    words_dict[item] = 1

            picture_line = '{} '.format(len(words_dict))
            for word in words_dict:
                picture_line = '{}{}:{} '.format(picture_line, word, words_dict[word])

            picture_line = picture_line[:-1] + '\n'
            f.write(picture_line)

if __name__ == '__main__':
    SEGMENT_PATH = './k_means_vocab/'
    ORB_LIST_OUTPATH = 'k_means_vecs/orb_vecs.pkl'
    XTRACT_LIST_OUTPATH = 'k_means_vecs/xtract_vecs.pkl'
    ORDER_OUTPATH = 'k_means_vecs/order.pkl'


    
    index_dict = dict()
    

    start_index = 0
    i = 0

    
    ORB_vecs_list = []

    XTRACT_vecs_list = []

    
    for folder in os.listdir(SEGMENT_PATH):
        if os.path.isdir('{}/{}'.format(SEGMENT_PATH, folder)):
            for file in os.listdir('{}/{}'.format(SEGMENT_PATH, folder)):
                if file[-3:] == 'jpg':
                    i += 1
                    ORB_vecs_list.append(extract_features_ORB('{}/{}/{}'.format(SEGMENT_PATH, folder, file)))
                    XTRACT_vecs_list.append(extract_features_XTRACT('{}/{}/{}'.format(SEGMENT_PATH, folder, file)))
                    print(i)
            index_dict[folder] = (start_index, i)
            start_index = i
    
    order_dict = dict()
    i = 0
    pres_dict = dict()
    j = 0
    for folder in os.listdir(SEGMENT_PATH):
        if os.path.isdir('{}/{}'.format(SEGMENT_PATH, folder)):
            for file in os.listdir('{}/{}'.format(SEGMENT_PATH, folder)):
                if file[-3:] == 'jpg':
                    pres_dict[j] = [folder, int(file.split('_')[1][:-4])]
                    j += 1

            order_dict[i] = folder
            i += 1
    
    
    with open(ORB_LIST_OUTPATH, 'wb') as f:
        pickle.dump(ORB_vecs_list, f)   

    with open(XTRACT_LIST_OUTPATH, 'wb') as f:
        pickle.dump(XTRACT_vecs_list, f)    
    
    with open(ORDER_OUTPATH, 'wb') as f:
        pickle.dump(order_dict, f)  
    
    
    #ORB_vecs_list = pickle.load(open(ORB_LIST_OUTPATH, 'rb'))
   #XTRACT_vecs_list = pickle.load(open(XTRACT_LIST_OUTPATH, 'rb'))
    HIST_vecs_list = pickle.load(open('k_means_vecs/hist_vecs.pkl', 'rb'))

    HS = []

    for i in range(0, len(ORB_vecs_list)):
        for item in HIST_vecs_list:
            if pres_dict[i][0] in item:
                HS.append(item[pres_dict[i][0]][pres_dict[i][1]])
    
    HIST_vecs_list = HS

    
    

    
    norm = ORB_vecs_list[0].shape
    mx = norm[0]
    for i in range(0, len(ORB_vecs_list)):
        item = ORB_vecs_list[i]
        if item.shape[0] > mx:
            mx = item.shape[0]

        

    needed_size = mx
    for i in range(0, len(ORB_vecs_list)):
        item = ORB_vecs_list[i]
        if item.size < needed_size:
   
            ORB_vecs_list[i] = np.concatenate([item, np.zeros(needed_size - item.size)])

    
    

    norm = ORB_vecs_list[0].shape
    mx = norm[0]
    for item in ORB_vecs_list:
        if item.shape[0] > mx:
            mx = item.shape[0]

        if item.shape != norm:
            print(item.shape)

    print('ready')
    
    km = KMeans(n_clusters = 50)
    ORB_clusters = km.fit_predict(ORB_vecs_list)
    print('c1')
    build_file(index_dict, ORB_clusters, 'k_means_dats/ORB_50_hlda_data.dat')
    print('b1') 
    km = KMeans(n_clusters = 50)
    XTRACT_clusters = km.fit_predict(XTRACT_vecs_list)
    print('c2')
    build_file(index_dict, XTRACT_clusters, 'k_means_dats/XTRACT_50_hlda_data.dat')
    print('b2')
    km = KMeans(n_clusters = 50)
    HIST_clusters = km.fit_predict(HIST_vecs_list)
    print('c3')
    build_file(index_dict, HIST_clusters, 'k_means_dats/HIST_50_hlda_data.dat')
    print('b3')
    

    km = KMeans(n_clusters = 20)
    ORB_clusters = km.fit_predict(ORB_vecs_list)
    print('c1')
    build_file(index_dict, ORB_clusters, 'k_means_dats/ORB_20_hlda_data.dat')
    print('b1') 
    km = KMeans(n_clusters = 20)
    XTRACT_clusters = km.fit_predict(XTRACT_vecs_list)
    print('c2')
    build_file(index_dict, XTRACT_clusters, 'k_means_dats/XTRACT_20_hlda_data.dat')
    print('b2')
    km = KMeans(n_clusters = 20)
    HIST_clusters = km.fit_predict(HIST_vecs_list)
    print('c3')
    build_file(index_dict, HIST_clusters, 'k_means_dats/HIST_20_hlda_data.dat')
    print('b3')
    
    km = KMeans(n_clusters = 30)
    ORB_clusters = km.fit_predict(ORB_vecs_list)
    print('c1')
    build_file(index_dict, ORB_clusters, 'k_means_dats/ORB_30_hlda_data.dat')
    print('b1') 
    km = KMeans(n_clusters = 30)
    XTRACT_clusters = km.fit_predict(XTRACT_vecs_list)
    print('c2')
    build_file(index_dict, XTRACT_clusters, 'k_means_dats/XTRACT_30_hlda_data.dat')
    print('b2')
    km = KMeans(n_clusters = 30)
    HIST_clusters = km.fit_predict(HIST_vecs_list)
    print('c3')
    build_file(index_dict, HIST_clusters, 'k_means_dats/HIST_30_hlda_data.dat')
    print('b3')
    
    km = KMeans(n_clusters = 100)
    ORB_clusters = km.fit_predict(ORB_vecs_list)
    print('c1')
    build_file(index_dict, ORB_clusters, 'k_means_dats/ORB_100_hlda_data.dat')
    print('b1') 
    km = KMeans(n_clusters = 100)
    XTRACT_clusters = km.fit_predict(XTRACT_vecs_list)
    print('c2')
    build_file(index_dict, XTRACT_clusters, 'k_means_dats/XTRACT_100_hlda_data.dat')
    print('b2')
    km = KMeans(n_clusters = 100)
    HIST_clusters = km.fit_predict(HIST_vecs_list)
    print('c3')
    build_file(index_dict, HIST_clusters, 'k_means_dats/HIST_100_hlda_data.dat')
    print('b3')
    
    
    
   


