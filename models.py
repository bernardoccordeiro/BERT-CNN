import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.contrib.slim as slim
import bert
import sys
import keras_metrics as km

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, Embedding, GlobalMaxPooling2D, Dropout, Reshape, Input, Concatenate
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.constraints import max_norm

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1"
FILTER_SIZES=[3, 4, 5]
NUM_FILTERS=5

def create_model_W2V(embedding_size, sentence_size, vocab_size, conv_n_filters, conv_filter_sizes, conv_window_size,
                 pool_window_size, hidden_layer_size, dropout_size, n_classes, embedding_weights=None, static=False,
                 rand=False, multichannel=False):

    inp = Input(shape=(sentence_size,))
    if multichannel:
        emb1 = Embedding(vocab_size, embedding_size, weights=[
                         embedding_weights], trainable=False)(inp)
        emb1 = Reshape((sentence_size, embedding_size, 1))(emb1)
        emb2 = Embedding(vocab_size, embedding_size, weights=[
                         embedding_weights], trainable=True)(inp)
        emb2 = Reshape((sentence_size, embedding_size, 1))(emb2)
        x = Concatenate()([emb1, emb2])
    else:
        if rand:
            emb = Embedding(vocab_size, embedding_size)(inp)
        elif static:
            emb = Embedding(vocab_size, embedding_size, weights=[
                            embedding_weights], trainable=False)(inp)
        else:
            emb = Embedding(vocab_size, embedding_size, weights=[
                            embedding_weights], trainable=True)(inp)
        x = Reshape((sentence_size, embedding_size, 1))(emb)

    convolution_layer = []
    for filter_size in conv_filter_sizes:
        conv_window_size = (filter_size, embedding_size)
        conv = Conv2D(conv_n_filters, conv_window_size,
                      activation='relu', use_bias=True, padding='valid')(x)
        convolution_layer.append(GlobalMaxPooling2D()(conv))

    x = Concatenate()(convolution_layer)
    x = Dropout(dropout_size)(x)
    x = Dense(n_classes, activation='softmax',
              kernel_constraint=max_norm(3))(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(optimizer=Adadelta(),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model


def create_model_BERT(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels, filter_sizes, num_filters, multichannel=False, static=False, print_summary=False):
    """Creates a classification model."""

    bert_inputs = dict(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids)

    if static:
        bert_module = hub.Module(
            BERT_MODEL_HUB,
            trainable=False)
    elif not static and not multichannel:
        bert_module = hub.Module(
            BERT_MODEL_HUB,
            trainable=True)
    elif multichannel:
        bert_module_nonstatic = hub.Module(
            BERT_MODEL_HUB,
            trainable=True)
        bert_module_static = hub.Module(
            BERT_MODEL_HUB,
            trainable=False)
    else:
        sys.exit('Model conf must be either STATIC, NONSTATIC or MULTICHANNEL!')

    if multichannel:
        bert_outputs_static = bert_module_static(
            inputs=bert_inputs,
            signature="tokens",
            as_dict=True)
        bert_outputs_nonstatic = bert_module_nonstatic(
            inputs=bert_inputs,
            signature="tokens",
            as_dict=True)

        output_layer_1 = bert_outputs_static["sequence_output"]
        output_layer_2 = bert_outputs_nonstatic["sequence_output"]
        embedding_size = output_layer_1.shape[-1].value
        sequence_length = tf.shape(output_layer_1)[1]
        output_layer_1 = tf.reshape(
            output_layer_1, [-1, sequence_length, embedding_size, 1])
        output_layer_2 = tf.reshape(
            output_layer_2, [-1, sequence_length, embedding_size, 1])
        output_layer = tf.concat([output_layer_1, output_layer_2], 3)
    if not multichannel:
        bert_outputs = bert_module(
            inputs=bert_inputs,
            signature="tokens",
            as_dict=True)

        output_layer = bert_outputs["sequence_output"]

        embedding_size = output_layer.shape[-1].value
        sequence_length = tf.shape(output_layer)[1]
        output_layer = tf.reshape(
            output_layer, [-1, sequence_length, embedding_size, 1])

  # Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    for filter_size in filter_sizes:
        with tf.name_scope("conv-maxpool-%s" % filter_size):
          # Convolution Layer
            if multichannel:
                filter_shape = [filter_size, embedding_size, 2, num_filters]
            else:
                filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(
                filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                output_layer,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.reduce_max(h, axis=1, keepdims=True)
            pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    output_weights = tf.get_variable(
        "output_weights", [num_labels, num_filters_total],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):

        # Dropout helps prevent overfitting
        h_pool_flat = tf.nn.dropout(h_pool_flat, keep_prob=0.5)

        logits = tf.matmul(h_pool_flat, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(
            tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
            return (predicted_labels, log_probs)

        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        def model_summary():
            model_vars = tf.trainable_variables()
            global_vars = tf.global_variables()
            print('Trainable variables:\n')
            slim.model_analyzer.analyze_vars(model_vars, print_info=True)
            print('All variables:\n')
            slim.model_analyzer.analyze_vars(global_vars, print_info=True)
        if print_summary:
            model_summary()
            sys.exit()

        return (loss, predicted_labels, log_probs)


def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps, multichannel=False, static=False, print_summary=False):
  def model_fn(features, labels, mode, params):
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

    # TRAIN and EVAL
    if not is_predicting:

      (loss, predicted_labels, log_probs) = create_model_BERT(
          is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels, filter_sizes=FILTER_SIZES, 
          num_filters=NUM_FILTERS, multichannel=multichannel, static=static, print_summary=print_summary)

      train_op = bert.optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

      # Calculate evaluation metrics.
      def metric_fn(label_ids, predicted_labels):
            accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
            recall = tf.metrics.recall(
                label_ids,
                predicted_labels)
            precision = tf.metrics.precision(
                label_ids,
                predicted_labels)
            return {
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision}

      eval_metrics = metric_fn(label_ids, predicted_labels)

      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)
      else:
          return tf.estimator.EstimatorSpec(mode=mode,
                                            loss=loss,
                                            eval_metric_ops=eval_metrics)
    else:
      (predicted_labels, log_probs) = create_model_BERT(
          is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels, filter_sizes=FILTER_SIZES, 
          num_filters=NUM_FILTERS, multichannel=multichannel, static=static)

      predictions = {
          'probabilities': log_probs,
          'labels': predicted_labels
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  return model_fn
