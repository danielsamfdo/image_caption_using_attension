import cPickle as pkl
import gzip
import os
import sys
import time

import numpy
import tables

def load_data(load_train=True, load_dev=True, load_test=True,
        path='data/flickr30k/'):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    #############
    # LOAD DATA #
    #############
    print '... loading data'

    if load_train:
        train_cap = pkl.load(open(path+'train.pkl', 'rb'))
        train_file = tables.open_file(path+'train-cnn_features.hdf5', mode='r')
        train_feat = train_file.root.feats[:]
        train = (train_cap, train_feat)
        print '... loaded train'
    else:
        train = None
    if load_test:
        test_cap = pkl.load(open(path+'test.pkl', 'rb'))
        test_file = tables.open_file(path+'test-cnn_features.hdf5', mode='r')
        test_feat = test_file.root.feats[:]
        test = (test_cap, test_feat)
        print '... loaded test'
    else:
        test = None
    if load_dev:
        dev_cap = pkl.load(open(path+'dev.pkl', 'rb'))
        dev_file = tables.open_file(path+'dev-cnn_features.hdf5', mode='r')
        dev_feat = dev_file.root.feats[:]
        valid = (dev_cap, dev_feat)
        print '... loaded dev'
    else:
        valid = None

    with open(path+'dictionary.pkl', 'rb') as f:
        worddict = pkl.load(f)

    return train, valid, test, worddict