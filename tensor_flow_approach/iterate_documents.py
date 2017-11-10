import logging
import os
import sys
from itertools import chain
from os import listdir
from os.path import isfile, join


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
