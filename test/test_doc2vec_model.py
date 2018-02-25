import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)

import os.path
import sys
from os.path import isfile, join

# gensim modules
from gensim import utils
from gensim.models import Doc2Vec

from gensim_approach import gensim_doc2vec_trainer


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logger.info("running %s" % ' '.join(sys.argv))

    directory_path = os.path.dirname(__file__)
    model_path = os.path.abspath(os.path.join(directory_path, '../resources/models/doc2vec-classifier-model.model'))
    model = Doc2Vec.load(model_path)

    test_data = os.path.abspath(os.path.join(directory_path, '../resources/paravec/do_prediction'))
    doc_data, doc_labels = gensim_doc2vec_trainer.read_documents(test_data)
    # print(doc_labels)

    # # # Inspecting the Model
    # # shows the similar words
    # print(model.most_similar('stock'))
    #
    # # shows the learnt embedding
    # print(model['stock'])
    #
    # # shows the similar docs with id = health
    # print(model.docvecs.most_similar("health"))
    #
    # for word in model.vocab.keys():
    #     print(word)

    # # Document Similarity
    for idx, doc in enumerate(doc_data):
        # print(doc)
        new_vector = model.infer_vector(doc)
        print(doc_labels[idx] + " ==> ")
        for item in model.docvecs.most_similar([new_vector]):
            print("\t" + str(item))
