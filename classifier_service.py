from flask import Flask, url_for, request
# from logging.config import thread
# import name_extractor
import json
import logging

from gensim_approach import gensim_doc2vec_trainer

import tensor_flow_approach.utils_ as i_docs
import tensor_flow_approach.text_classify_rnn as classifier_rnn
import tensor_flow_approach.text_classify_cnn as classifier_cnn

application = Flask(__name__)


@application.route('/tfcategorizer', methods=['POST'])
def tfcategorizer():
    try:
        if request.method == 'POST':
            return None

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
