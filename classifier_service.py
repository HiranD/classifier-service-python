from flask import Flask, url_for, request
# from logging.config import thread
# import name_extractor
import json
import logging

application = Flask(__name__)


@application.route('/tfcategorizer', methods=['POST'])
def tfcategorizer():
    try:
        if request.method == 'POST':
            return name_extractor.getNames(request.form['content'])

    except Exception as e:
        logging.error("error: " + str(e))


@application.route('/gensimcategorizer', methods=['POST'])
def gensimcategorizer():
    try:
        if request.method == 'POST':
            return name_extractor.getNames(request.form['content'])

    except Exception as e:
        logging.error("error: " + str(e))


if __name__ == '__main__':
    application.run(threaded=True)
