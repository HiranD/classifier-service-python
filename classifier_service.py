from flask import Flask, url_for, request
# from logging.config import thread
# import name_extractor
import json
import logging
import argparse
import os.path

from gensim_approach import gensim_doc2vec_trainer

import tensor_flow_approach.utils_ as i_docs
import tensor_flow_approach.text_classify_rnn as classifier_rnn
import tensor_flow_approach.text_classify_cnn as classifier_cnn

application = Flask(__name__)


@application.route('/tfcategorizer', methods=['POST'])
def tfcategorizer():
    try:
        directory_path = os.path.dirname(__file__)
        model_path = os.path.abspath(os.path.join(directory_path, 'resources/models/tf_rnn/'))
        FLAGS = argparse.Namespace()
        FLAGS.bow_model = False
        prediction_data = []
        if request.method == 'POST':
            prediction_data.append(request.get_data(as_text=True))
            return classifier_rnn.classify(FLAGS, prediction_data, model_path)

    except Exception as e:
        logging.error("error: " + str(e))


@application.route('/gensimcategorizer', methods=['POST'])
def gensimcategorizer():
    try:
        if request.method == 'POST':
            return None

    except Exception as e:
        logging.error("error: " + str(e))


if __name__ == '__main__':
    application.run(threaded=True)
