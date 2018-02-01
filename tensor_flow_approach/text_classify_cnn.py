from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas
import tensorflow as tf
import tensor_flow_approach.utils_ as utils_

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_DOCUMENT_LENGTH = 100
EMBEDDING_SIZE = 20
N_FILTERS = 10
WINDOW_SIZE = 20
FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]
FILTER_SHAPE2 = [WINDOW_SIZE, N_FILTERS]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
n_words = 0
MAX_LABEL = 15
WORDS_FEATURE = 'words'  # Name of the input words feature.
vocab_processor = None


def cnn_model(features, labels, mode):
    """2 layer ConvNet to predict from sequence of words to a class."""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = tf.contrib.layers.embed_sequence(
        features[WORDS_FEATURE], vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
    word_vectors = tf.expand_dims(word_vectors, 3)
    with tf.variable_scope('CNN_Layer1'):
        # Apply Convolution filtering on input sequence.
        conv1 = tf.layers.conv2d(
            word_vectors,
            filters=N_FILTERS,
            kernel_size=FILTER_SHAPE1,
            padding='VALID',
            # Add a ReLU for non linearity.
            activation=tf.nn.relu)
        # Max pooling across output of Convolution+Relu.
        pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=POOLING_WINDOW,
            strides=POOLING_STRIDE,
            padding='SAME')
        # Transpose matrix so that n_filters from convolution becomes width.
        pool1 = tf.transpose(pool1, [0, 1, 3, 2])
    with tf.variable_scope('CNN_Layer2'):
        # Second level of convolution filtering.
        conv2 = tf.layers.conv2d(
            pool1,
            filters=N_FILTERS,
            kernel_size=FILTER_SHAPE2,
            padding='VALID')
        # Max across each filter to get useful features for classification.
        pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])

    # Apply regular WX + B and classification.
    logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)

    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'class': predicted_classes,
                'prob': tf.nn.softmax(logits)
            })

    onehot_labels = tf.one_hot(labels, MAX_LABEL, 1, 0)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predicted_classes)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(FLAGS, train_data, train_labels, test_data, test_labels, model_dir):
    global n_words

    # Prepare training and testing data
    labels = pandas.factorize(train_labels + test_labels)[0]
    labels_ = pandas.factorize(train_labels + test_labels)[1]

    x_train = pandas.Series(train_data)
    y_train = pandas.Series(labels[:len(train_labels)])
    x_test = pandas.Series(test_data)
    y_test = pandas.Series(labels[-len(test_labels):])

    # Process vocabulary
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        MAX_DOCUMENT_LENGTH)
    x_transform_train = vocab_processor.fit_transform(x_train)
    x_transform_test = vocab_processor.transform(x_test)

    x_train = np.array(list(x_transform_train))
    x_test = np.array(list(x_transform_test))

    n_words = len(vocab_processor.vocabulary_)
    logging.info('Total words: %d' % n_words)

    vocab_processor.save(model_dir + "/vocabulary_")

    utils_.save_factorization(labels_, model_dir)

    # Build model
    classifier = tf.estimator.Estimator(model_fn=cnn_model, model_dir=model_dir)

    # Train.
    if FLAGS.train:
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={WORDS_FEATURE: x_train},
            y=y_train,
            batch_size=len(x_train),
            num_epochs=None,
            shuffle=True)
        classifier.train(input_fn=train_input_fn, steps=100)

    # Evaluate.
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={WORDS_FEATURE: x_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)

    scores = classifier.evaluate(input_fn=test_input_fn)
    logging.info('Accuracy: {0:f}'.format(scores['accuracy']))

    predictions = classifier.predict(input_fn=test_input_fn)
    y_predicted = np.array(list(p['class'] for p in predictions))
    y_predicted = y_predicted.reshape(np.array(y_test).shape)

    labels_ = utils_.load_factorization(model_dir)
    labels = list(labels_[x] for x in y_predicted)
    logging.info('Predicted categories: [' + ', '.join(labels) + "]")


def load_classifier(FLAGS, model_dir):
    global vocab_processor
    global n_words
    tf.logging.set_verbosity(tf.logging.INFO)

    # Process vocabulary
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        MAX_DOCUMENT_LENGTH).restore(model_dir + "/vocabulary_")
    n_words = len(vocab_processor.vocabulary_)

    logging.info('Total words: %d' % n_words)

    # Build model
    classifier = tf.estimator.Estimator(model_fn=cnn_model, model_dir=model_dir)
    return classifier


def classify(FLAGS, test_data, model_dir, classifier):
    global vocab_processor
    tf.logging.set_verbosity(tf.logging.INFO)
    x_test = pandas.Series(test_data)
    x_transform_test = vocab_processor.transform(x_test)

    x_test = np.array(list(x_transform_test))

    # Evaluate.
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={WORDS_FEATURE: x_test},
        num_epochs=1,
        shuffle=False)

    predictions = classifier.predict(input_fn=test_input_fn)
    y_predicted = list(p['class'] for p in predictions)
    labels_ = utils_.load_factorization(model_dir)
    labels = list(labels_[x] for x in y_predicted)

    return labels[0]
