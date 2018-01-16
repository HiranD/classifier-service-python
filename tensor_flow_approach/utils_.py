import logging
import os
import sys
from itertools import chain
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas


def list_directory(source):
    return [f for f in listdir(source)]


def list_documents(source):
    return [f for f in listdir(source) if isfile(join(source, f))]


def read_documents(dir_path):
    doc_labels = []
    doc_data = []
    for directory in list_directory(dir_path):
        for doc in list_documents(os.path.join(dir_path, directory)):
            with open(os.path.join(dir_path, os.path.join(directory, doc)), "r") as f:
                doc_data.append(f.read())
                doc_labels.append(directory)
    return doc_data, doc_labels


def get_documents(relative_path):
    directory_path = os.path.dirname(__file__)
    dir_path = os.path.abspath(os.path.join(directory_path, relative_path))

    doc_data, doc_labels = read_documents(dir_path)
    return doc_data, doc_labels


def save_training_data(data, labels, relative_path):
    directory_path = os.path.dirname(__file__)
    dir_path = os.path.abspath(os.path.join(directory_path, relative_path))

    np.savetxt(dir_path + '/training_data_.npy', data, fmt='%d')
    labels.to_csv(dir_path + '/training_labels_.csv')


def load_training_data(relative_path):
    directory_path = os.path.dirname(__file__)
    dir_path = os.path.abspath(os.path.join(directory_path, relative_path))

    data = np.loadtxt(dir_path + '/training_data_.npy', dtype=int)
    labels = pandas.Series.from_csv(dir_path + '/training_labels_.csv')
    return data, labels


def save_factorization(labels_, relative_path):
    directory_path = os.path.dirname(__file__)
    dir_path = os.path.abspath(os.path.join(directory_path, relative_path))

    np.savetxt(dir_path + '/factorized_labels_.npy', labels_, fmt='%s')


def load_factorization(relative_path):
    directory_path = os.path.dirname(__file__)
    dir_path = os.path.abspath(os.path.join(directory_path, relative_path))

    labels = np.loadtxt(dir_path + '/factorized_labels_.npy', dtype=str)
    return labels
