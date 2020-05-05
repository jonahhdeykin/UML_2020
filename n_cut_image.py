
from skimage import data, segmentation, color, io, img_as_ubyte
from skimage.exposure import histogram
import matplotlib.pyplot as plt
from skimage.future import graph
import numpy as np
import pickle

import os

def create_mask(filename, n_segments=400, n_cuts=10):
    img = io.imread(filename)

    if img.shape[2] == 4:
        img = img_as_ubyte(color.rgba2rgb(img))

    labels1 = segmentation.slic(img, n_segments=n_segments)
    
    
    rag = graph.rag_mean_color(img, labels1, mode='similarity')
        
    labels2 = graph.cut_normalized(labels1, rag, num_cuts=n_cuts)

    return labels2, img

def save_segments(labels, img, image_name, mask=-1):
    try:
        os.mkdir('n_cut_vocabulary/'+image_name)
    except:
        pass

    for value in np.unique(labels):
        print(labels.shape)
        print(img.shape)
        segment = np.where(np.expand_dims(labels, axis=-1)==value, img, [0, 0, 0])
        # This creates a white-black image
        # temp_out = color.label2rgb(temp_label, img, bg_label=mask, colors=['white', 'black'], image_alpha=0, alpha=1)

        # This overlays on the average of background
        #temp_out = color.label2rgb(temp_label, img)
        segment = segment.astype(np.single)
        print('saving {}/mask_{}'.format(image_name, value))
        # plt.imsave('n_cut_vocabulary/' + image_name + '/mask_{}'.format(value) + '.jpg', segment.astype(np.uint8))


if __name__ == "__main__":
    PICTURE_PATH = './Final_Dataset'
    current_file = 0
    total_files = len(os.listdir(PICTURE_PATH))
    hist_list = []
    k = 0

    for file in os.listdir(PICTURE_PATH):

        if file[-3:] == 'jpg':

            print('{}/{}'.format(PICTURE_PATH, file))
            
            labels, img = create_mask('{}/{}'.format(PICTURE_PATH, file))
            image_name = file[:-4]
            # Each image has a dictionary
            image_dict = {image_name: {}}
    
            for j in np.unique(labels):
                a, b = np.where(labels == j)
                new_count = np.zeros((86))

                for i in range(0, len(a)):

                    avg = (img[a[i]][b[i]][0] + img[a[i]][b[i]][1] + img[a[i]][b[i]][2])/3
                    bucket = int(avg)
                    
                    new_count[bucket] += 1

                # Add segement for the current image name gets a count vector
                image_dict[image_name][j] = new_count

            hist_list.append(image_dict)
            

            #turn this on to save the resulting image segments
            save_segments(labels, img, image_name)
            
            #print(io.imread('{}/{}'.format(PICTURE_PATH, file)).shape)
            #print(file)
            
        current_file += 1
        print(current_file, total_files)
    
    with open('hist_vecs.pkl', 'wb') as f:
        pickle.dump(hist_list, f)

