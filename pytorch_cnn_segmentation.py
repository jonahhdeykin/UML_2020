# Found at this link: https://github.com/kanezaki/pytorch-unsupervised-segmentation
# Started at 7:04

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
from skimage import segmentation
import torch.nn.init
import matplotlib.pyplot as plt
import os
import pickle
from skimage import data, segmentation, color, io, img_as_ubyte
import shutil

use_cuda = torch.cuda.is_available()


# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim, nChannel=100, nConv=2):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(nConv-1):
            self.conv2.append( nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(nChannel) )
        self.conv3 = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(nChannel)

    def forward(self, x, nConv=2):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

def load_image(fileinput):
    # load image
    im = io.imread(fileinput)
    if im.shape[2] == 4:
        im = img_as_ubyte(color.rgba2rgb(im))

    data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) )
    if use_cuda:
        data = data.cuda()
    data = Variable(data)

    return im, data

def slice_and_train_model(im, data, nChannel=100, maxIter=15, minLabels=2, lr=.1,
                          num_superpixels=400):
    # slic
    labels = segmentation.slic(im, n_segments=num_superpixels)
    labels = labels.reshape(im.shape[0]*im.shape[1])
    u_labels = np.unique(labels)
    l_inds = []
    for i in range(len(u_labels)):
        l_inds.append( np.where( labels == u_labels[ i ] )[ 0 ] )

    # train
    model = MyNet( data.size(1) )
    if use_cuda:
        model.cuda()
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for batch_idx in range(maxIter):
        # forwarding
        optimizer.zero_grad()
        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )
        ignore, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))

        # superpixel refinement
        # TODO: use Torch Variable instead of numpy for faster calculation
        for i in range(len(l_inds)):
            labels_per_sp = im_target[ l_inds[ i ] ]
            u_labels_per_sp = np.unique( labels_per_sp )
            hist = np.zeros( len(u_labels_per_sp) )
            for j in range(len(hist)):
                hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
            im_target[ l_inds[ i ] ] = u_labels_per_sp[ np.argmax( hist ) ]
        target = torch.from_numpy( im_target )
        if use_cuda:
            target = target.cuda()
        target = Variable( target )
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        #print (batch_idx, '/', args.maxIter, ':', nLabels, loss.data[0])
        print (batch_idx, '/', maxIter, ':', nLabels, loss.item())

        if nLabels <= minLabels:
            print ("nLabels", nLabels, "reached minLabels", minLabels, ".")
            raise Exception("Not enough labels")

    return im, im_target

def save_segments(im_target, im, image_name, mask=-1):

    if os.path.exists('pytorch_cnn_vocab/' + image_name):
        try:
            shutil.rmtree('pytorch_cnn_vocab/' + image_name)
            print("Deleted {}".format('pytorch_cnn_vocab/' + image_name))
        except OSError as e:
            print("Error: %s : %s" % ('pytorch_cnn_vocab/' + image_name, e.strerror))

    try:
        os.mkdir('pytorch_cnn_vocab/' + image_name)
    except:
        pass

    for value in np.unique(im_target):
        segment = np.where(np.reshape(im_target == value, (im.shape[0], im.shape[1], 1)), im, [0, 0, 0])

        # This overlays on the average of background
        segment = segment.astype(np.single)
        print('saving {}/mask_{}'.format(image_name, value))
        plt.imsave('pytorch_cnn_vocab/' + image_name + '/mask_{}'.format(value) + '.jpg',
                   segment.astype(np.uint8))



if __name__=="__main__":
    PICTURE_PATH = './Final_Dataset_300'
    current_file = 0
    total_files = len(os.listdir(PICTURE_PATH))
    hist_list = []
    k = 0

    for file in os.listdir(PICTURE_PATH):

        if file[-3:] == 'jpg':

            print('{}/{}'.format(PICTURE_PATH, file))
            image_name = file[:-4]
            # Each image has a dictionary
            image_dict = {image_name: {}}

            im, data = load_image('{}/{}'.format(PICTURE_PATH, file))

            try:
                im, im_target = slice_and_train_model(im, data)

                flat_image = np.reshape(im, (im.shape[0] * im.shape[1], 3))
                for j in np.unique(im_target):
                    indices = np.where(im_target == j)
                    indices = indices[0]
                    # indices = np.reshape(im_target == j, (im.shape[0] * im.shape[1], 1))
                    new_count = np.zeros((86))

                    for i in range(0, len(indices)):
                        avg = (flat_image[indices[i]][0] + flat_image[indices[i]][1] + flat_image[indices[i]][2]) / 3
                        bucket = int(avg)

                        new_count[bucket] += 1

                    # Add segement for the current image name gets a count vector
                    image_dict[image_name][j] = new_count

                hist_list.append(image_dict)

                save_segments(im_target, im, image_name)

            except:
                print("Image: " + image_name + " did not have enough labels.")

        current_file += 1
        print(current_file, total_files)

    with open('pytorch_cnn_hist_vecs.pkl', 'wb') as f:
        pickle.dump(hist_list, f)