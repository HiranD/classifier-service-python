import logging
import os.path
import sys
import argparse

import tensor_flow_approach.iterate_documents as i_docs
import tensor_flow_approach.text_classify_rnn as classifier_rnn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bow_model',
        default=False,
        help='Run with BOW model instead of RNN.',
        action='store_true')
    FLAGS, unparsed = parser.parse_known_args()

    train_data, train_labels = i_docs.get_documents('../resources/paravec/labeled')
    test_data, test_labels = i_docs.get_documents('../resources/paravec/unlabeled')

    directory_path = os.path.dirname(__file__)
    model_path = os.path.abspath(os.path.join(directory_path, '../resources/models/tf_rnn/'))

    classifier_rnn.main(FLAGS, train_data, train_labels, test_data, test_labels, model_path)
