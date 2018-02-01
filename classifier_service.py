from flask import Flask, url_for, request
# from logging.config import thread
# import name_extractor
import json
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse
import os.path

from gensim_approach import gensim_doc2vec_trainer

import tensor_flow_approach.utils_ as i_docs
import tensor_flow_approach.text_classify_rnn as classifier_rnn
import tensor_flow_approach.text_classify_cnn as classifier_cnn

application = Flask(__name__)
directory_path = os.path.dirname(__file__)
FLAGS = argparse.Namespace()
MODEL = None
model_path = None
CLASSIFIER = None
RNN = False
CNN = False
previous_model = [RNN, CNN]


# @application.before_first_request
# def _run_on_start():
#     global CLASSIFIER
#     model_path = os.path.abspath(os.path.join(directory_path, 'resources/models/tf_rnn/'))
#     FLAGS.bow_model = False
#     logging.info("loading RNN the model..")
#     CLASSIFIER = classifier_rnn.load_classifier(FLAGS, model_path)
#     logging.info("loading the model has finished..")


def _run_on_first_request():
    global MODEL
    global CLASSIFIER
    global previous_model
    global model_path

    if previous_model != [RNN, CNN]:
        if RNN:
            logging.info("loading RNN the model..")
            FLAGS.bow_model = False
            model_path = os.path.abspath(os.path.join(directory_path, 'resources/models/tf_rnn/'))
            CLASSIFIER = classifier_rnn
            previous_model = [RNN, CNN]
        if CNN:
            logging.info("loading CNN the model..")
            model_path = os.path.abspath(os.path.join(directory_path, 'resources/models/tf_cnn/'))
            CLASSIFIER = classifier_cnn
            previous_model = [RNN, CNN]

        MODEL = CLASSIFIER.load_classifier(FLAGS, model_path)
        logging.info("loading the model has finished..")
    return True


@application.route('/tfcategorizer', methods=['POST'])
def tfcategorizer():
    global RNN
    global CNN
    if request.headers.get('model') == "CNN":
        RNN = False
        CNN = True
    else:
        RNN = True
        CNN = False

    if _run_on_first_request():
        try:
            prediction_data = []
            if request.method == 'POST':
                prediction_data.append(request.get_data(as_text=True))
                return CLASSIFIER.classify(FLAGS, prediction_data, model_path, MODEL)
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
    # _run_on_start()
    application.run(threaded=True)
