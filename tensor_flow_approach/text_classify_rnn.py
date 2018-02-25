from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import tensor_flow_approach.utils_ as utils_

MAX_DOCUMENT_LENGTH = 10
EMBEDDING_SIZE = 50
n_words = 0
MAX_LABEL = 15
WORDS_FEATURE = 'words'  # Name of the input words feature.
vocab_processor = None


def estimator_spec_for_softmax_classification(
        logits, labels, mode):
    """Returns EstimatorSpec instance for softmax classification."""
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


def bag_of_words_model(features, labels, mode):
    """A bag-of-words model. Note it disregards the word order in the text."""
    bow_column = tf.feature_column.categorical_column_with_identity(
        WORDS_FEATURE, num_buckets=n_words)
    bow_embedding_column = tf.feature_column.embedding_column(
        bow_column, dimension=EMBEDDING_SIZE)
    bow = tf.feature_column.input_layer(
        features,
        feature_columns=[bow_embedding_column])
    logits = tf.layers.dense(bow, MAX_LABEL, activation=None)

    return estimator_spec_for_softmax_classification(
        logits=logits, labels=labels, mode=mode)


def rnn_model(features, labels, mode):
    """RNN model to predict from sequence of words to a class."""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = tf.contrib.layers.embed_sequence(
        features[WORDS_FEATURE], vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

    # Split into list of embedding per word, while removing doc length dim.
    # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
    word_list = tf.unstack(word_vectors, axis=1)

    # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
    cell = tf.nn.rnn_cell.GRUCell(EMBEDDING_SIZE)

    # Create an unrolled Recurrent Neural Networks to length of
    # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

    # Given encoding of RNN, take encoding of last step (e.g hidden size of the
    # neural network of last step) and pass it as features for softmax
    # classification over output classes.
    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
    return estimator_spec_for_softmax_classification(
        logits=logits, labels=labels, mode=mode)


def main(FLAGS, train_data, train_labels, test_data, test_labels, model_dir):
    global n_words
    tf.logging.set_verbosity(tf.logging.INFO)

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

    vocab_processor.save(model_dir+"/vocabulary_")
    # print(x_train)
    # print(y_train)
    # utils_.save_training_data(x_train, y_train, model_dir)
    utils_.save_factorization(labels_, model_dir)

    # Build model
    # Switch between rnn_model and bag_of_words_model to test different models.
    model_fn = rnn_model
    if FLAGS.bow_model:
        # Subtract 1 because VocabularyProcessor outputs a word-id matrix where word
        # ids start from 1 and 0 means 'no word'. But
        # categorical_column_with_identity assumes 0-based count and uses -1 for
        # missing word.
        x_train -= 1
        x_test -= 1
        model_fn = bag_of_words_model
    classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir)

    # Train.
    if FLAGS.train:
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={WORDS_FEATURE: x_train},
            y=y_train,
            batch_size=len(x_train),
            num_epochs=None,
            shuffle=True)
        classifier.train(input_fn=train_input_fn, steps=100)

    # Predict.
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={WORDS_FEATURE: x_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)
    predictions = classifier.predict(input_fn=test_input_fn)
    y_predicted = np.array(list(p['class'] for p in predictions))
    y_predicted = y_predicted.reshape(np.array(y_test).shape)

    labels_ = utils_.load_factorization(model_dir)
    labels = list(labels_[x] for x in y_predicted)
    logging.info('Predicted categories: [' + ', '.join(labels) + "]")

    # Score with sklearn.
    score = metrics.accuracy_score(y_test, y_predicted)
    logging.info('Accuracy (sklearn): {0:f}'.format(score))

    # Score with tensorflow.
    scores = classifier.evaluate(input_fn=test_input_fn)
    logging.info('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))


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
    # Switch between rnn_model and bag_of_words_model to test different models.
    model_fn = rnn_model
    if FLAGS.bow_model:
        # Subtract 1 because VocabularyProcessor outputs a word-id matrix where word
        # ids start from 1 and 0 means 'no word'. But
        # categorical_column_with_identity assumes 0-based count and uses -1 for
        # missing word.
        model_fn = bag_of_words_model
    classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir)
    return classifier


def classify(FLAGS, test_data, model_dir, classifier):
    global vocab_processor
    tf.logging.set_verbosity(tf.logging.INFO)
    x_test = pandas.Series(test_data)
    x_transform_test = vocab_processor.transform(x_test)

    x_test = np.array(list(x_transform_test))

    if FLAGS.bow_model:
        # Subtract 1 because VocabularyProcessor outputs a word-id matrix where word
        # ids start from 1 and 0 means 'no word'. But
        # categorical_column_with_identity assumes 0-based count and uses -1 for
        # missing word.
        x_test -= 1

    # Predict.
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={WORDS_FEATURE: x_test},
        num_epochs=1,
        shuffle=False)

    predictions = classifier.predict(input_fn=test_input_fn)
    y_predicted = list(p['class'] for p in predictions)
    labels_ = utils_.load_factorization(model_dir)
    labels = list(labels_[x] for x in y_predicted)

    return labels
