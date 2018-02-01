import logging
import os.path
import sys
import argparse

import time

import tensor_flow_approach.utils_ as i_docs
import tensor_flow_approach.text_classify_cnn as classifier_cnn

if __name__ == '__main__':
    train_data, train_labels = i_docs.get_documents('../resources/paravec/labeled')
    test_data, test_labels = i_docs.get_documents('../resources/paravec/unlabeled')
    prediction_data, prediction_labels = i_docs.get_documents('../resources/paravec/do_prediction')

    start_time = time.time()
    directory_path = os.path.dirname(__file__)
    model_path = os.path.abspath(os.path.join(directory_path, '../resources/models/tf_cnn/'))

    FLAGS = argparse.Namespace()

    # FLAGS.train = True
    # classifier_cnn.main(FLAGS, train_data, train_labels, test_data, test_labels, model_path)

    print("------------------------------------------------------------")

    # FLAGS.train = False
    # classifier_cnn.main(FLAGS, train_data, train_labels, prediction_data, prediction_labels, model_path)

    print(classifier_cnn.classify(FLAGS, prediction_data, model_path, classifier_cnn.load_classifier(FLAGS, model_path)))

    print("--- %s seconds ---" % (time.time() - start_time))
    print(model_path)
    print(prediction_labels)
