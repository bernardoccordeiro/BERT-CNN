from preprocessing import clean_str
import numpy as np
import pandas as pd
import os

def load_mr():
    train_neg = []
    train_pos = []
    with open('rt-polaritydata/rt-polarity.neg', 'rb') as neg_r:
        line = neg_r.readline()
        while line:
            train_neg.append(clean_str(str(line)))
            line = neg_r.readline()
    with open('rt-polaritydata/rt-polarity.pos', 'rb') as pos_r:
        line = pos_r.readline()
        while line:
            train_pos.append(clean_str(str(line)))
            line = pos_r.readline()
    train_data = np.concatenate((train_pos, train_neg))

    pos_labels = np.ones(len(train_pos))
    neg_labels = np.zeros(len(train_neg))
    labels = np.concatenate((pos_labels, neg_labels))
    
    return train_data, labels

def load_trec(int_labels=True, get_test_set=True):
    train_labels = []
    train_data = []
    labels_dict = {}
    label_counter = 0

    with open('TREC-data/train_5500.label', 'rb') as data_file:
        for line in data_file.readlines():
            label_specific, sentence = str(line).split(' ', 1)
            label = label_specific.split(':')[0][2:]
            sentence = sentence.replace('\n', '')
            if int_labels:
                if label in labels_dict:
                    label = labels_dict[label]
                else:
                    labels_dict[label] = label_counter
                    label = label_counter
                    label_counter += 1
            train_labels.append(label)
            train_data.append(sentence)
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    test_labels = []
    test_data = []
    with open('TREC-data/TREC_10.label', 'rb') as test_file:
        for line in test_file.readlines():
            label_specific, sentence = str(line).split(' ', 1)
            label = label_specific.split(':')[0][2:]
            sentence = sentence.replace('\n', '')
            if int_labels:
                label = labels_dict[label]
            test_labels.append(label)
            test_data.append(sentence)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    if get_test_set:
        return train_data, train_labels, test_data, test_labels
    else:
        return train_data, labels

def load_portuguese_twitter():
    data = pd.read_csv('PortTwitter/unbalanced_corpus/corpus_three_class_unbalanced/tweets-total-csv-3-class-unbalanced.csv', 
        sep=';', encoding='latin')
    train_data = data['tweet'].values
    labels = pd.factorize(data['OPINIAO'])[0]

    return train_data, labels
