import logging
import os
import sys
import multiprocessing
from itertools import chain
from os import listdir
from os.path import isfile, join
import gensim
from gensim import utils
from nltk.corpus import stopwords
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from nltk import RegexpTokenizer, PorterStemmer, re
from sklearn.linear_model import LogisticRegression


def dir_data_flow(data, labels):
    for i, j in zip(data, labels):
        yield TaggedDocument(i, [j])


def list_directory(source):
    return [f for f in listdir(source)]


def list_documents(source):
    return [f for f in listdir(source) if isfile(join(source, f))]


def prepare_document(text):
    tokenizer = RegexpTokenizer(r'\w+')

    raw = text.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    filtered_tokens = [i for i in tokens if i not in stopwords.words('english')]

    # remove numbers
    number_tokens = [re.sub(r'[\d]', ' ', i) for i in filtered_tokens]
    number_tokens = ' '.join(number_tokens).split()

    # stem tokens
    stemmed_tokens = [PorterStemmer().stem(i) for i in number_tokens]

    # remove empty
    length_tokens = [i for i in stemmed_tokens if len(i) > 1]

    return length_tokens


def read_documents(dir_path):
    doc_labels = []
    doc_data = []
    for directory in list_directory(dir_path):
        for doc in list_documents(os.path.join(dir_path, directory)):
            with open(os.path.join(dir_path, os.path.join(directory, doc)), "r") as f:
                doc_data.append(prepare_document(f.read()))
                doc_labels.append(directory)
    return doc_data, doc_labels


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    directory_path = os.path.dirname(__file__)
    dataDirPath = os.path.abspath(os.path.join(directory_path, '../resources/data/labeled'))
    model_path = os.path.abspath(os.path.join(directory_path, '../resources/models'))

    doc_data, doc_labels = read_documents(dataDirPath)
    print(doc_data)
    print(doc_labels)

    tagged_documents = dir_data_flow(doc_data, doc_labels)
    model = Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=10, hs=0, min_count=2,
                    workers=multiprocessing.cpu_count())
    model.build_vocab(tagged_documents)

    epochs = 50
    for epoch in range(epochs):
        model.train(tagged_documents)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no deca
        model.train(tagged_documents)

    model.save(model_path + "/doc2vec-classifier-model.model")


