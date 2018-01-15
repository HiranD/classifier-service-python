import logging
import os.path
import sys

import tensor_flow_approach.utils_ as utils
import tensor_flow_approach.text_classify_cnn as classifier_cnn

if __name__ == '__main__':
  train_data, train_labels = utils.get_documents('../resources/paravec/labeled')
  test_data, test_labels = utils.get_documents('../resources/paravec/unlabeled')

  directory_path = os.path.dirname(__file__)
  model_path = os.path.abspath(os.path.join(directory_path, '../resources/models/tf_cnn/'))

  classifier_cnn.main(train_data, train_labels, test_data, test_labels, model_path)
