import load_dataset as dataset
import numpy as np
import tensorflow as tf
import pickle
import sys
import os

from datetime import datetime
from tensorflow import set_random_seed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard
from keras.utils import to_categorical
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from models import create_model_W2V

np.random.RandomState(42)
set_random_seed(2)

if len(sys.argv) != 2:
    sys.exit('Please provide a dataset name!')

OUTPUT_DIR_W2V = 'output_w2v'
DATASET_NAME = sys.argv[1]
EMBEDDING_SIZE = 300
CONV_N_FILTERS = 5
CONV_FILTER_SIZES = [3, 4, 5]
CONV_WINDOW_SIZE = (5, EMBEDDING_SIZE)
POOL_WINDOW_SIZE = (4, 4)
HIDDEN_LAYER_SIZE = 32
DROPOUT_SIZE = 0.5
STATIC = False
RAND = False
MULTICHANNEL = True

BATCH_SIZE = 8
N_EPOCHS = 10
KFOLD_SIZE = 10

tensorboard_logdir = "{}/log_{}".format(OUTPUT_DIR_W2V, DATASET_NAME)
os.makedirs(tensorboard_logdir, exist_ok=True)

test_data = []
test_labels = []
if DATASET_NAME == 'MR':
    print('Loading MR Dataset')
    train_data, labels = dataset.load_mr()
elif DATASET_NAME == 'TREC':
    print('Loading TREC Dataset')
    train_data, labels, test_data, test_labels = dataset.load_trec()
elif DATASET_NAME == 'PORT_TWITTER':
    print('Loading Portuguese Twitter Dataset')
    train_data, labels = dataset.load_portuguese_twitter()
else:
    sys.exit('DATASET_NAME is not a valid dataset!')

if len(test_data) > 0:
    has_test_data = True
else:
    has_test_data = False


shuffled_indices = list(range(len(train_data)))
np.random.shuffle(shuffled_indices)

train_data = train_data[shuffled_indices]
labels = labels[shuffled_indices]
n_classes = len(np.unique(labels))


# Tokenization and label preprocessing
labels = to_categorical(labels)

print('Transforming tokens to indexes')
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
tokenizer.fit_on_texts(train_data)

word_index = tokenizer.word_index
VOCAB_SIZE = len(word_index) + 1

X_train = tokenizer.texts_to_sequences(train_data)
SENTENCE_SIZE = max(list(map(lambda x: len(x), X_train)))
X_train = pad_sequences(X_train, maxlen=SENTENCE_SIZE, padding='post')

if has_test_data:
    test_labels = to_categorical(test_labels)
    X_test = tokenizer.texts_to_sequences(test_data)
    X_test = pad_sequences(X_test, maxlen=SENTENCE_SIZE, padding='post')

reverse_word_index = {v: k for k, v in word_index.items()}
reverse_word_index[0] = '<PAD>'

# Word2Vec
print('Loading Word2Vec')
w2v = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin', binary=True)

print('Creating embedding weights')
embedding_weights = []
embedding_weights.append(list(np.zeros(EMBEDDING_SIZE)))
for k, v in word_index.items():
    try:
        wv = w2v[k]
    except:
        wv = np.random.random(EMBEDDING_SIZE)
    embedding_weights.append(list(wv))
embedding_weights = np.array(embedding_weights)

model_names = ['MULTICHANNEL', 'STATIC', 'NONSTATIC', 'RAND']
model_confs = [
    {'static': False, 'rand': False, 'multichannel': True},
    {'static': True, 'rand': False, 'multichannel': False},
    {'static': False, 'rand': False, 'multichannel': False},
    {'static': False, 'rand': True, 'multichannel': False}
]


for model_name, model_conf in zip(model_names, model_confs):
    print('Training {} model'.format(model_name))
    model = create_model_W2V(EMBEDDING_SIZE, SENTENCE_SIZE, VOCAB_SIZE, conv_n_filters=CONV_N_FILTERS, conv_filter_sizes=CONV_FILTER_SIZES,
                             conv_window_size=CONV_WINDOW_SIZE, pool_window_size=POOL_WINDOW_SIZE, hidden_layer_size=HIDDEN_LAYER_SIZE, dropout_size=DROPOUT_SIZE,
                             n_classes=n_classes, embedding_weights=embedding_weights, static=model_conf['static'], rand=model_conf['rand'], multichannel=model_conf['multichannel'])
    # print(model_name)
    # model.summary()
    # continue

    model_histories = []
    training_times = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_scores = []
    confusion_matrices = []
    predictions = []

    if not has_test_data:
        print('Dataset has no standard train/test split. Performing {}-Fold CV'.format(KFOLD_SIZE))
        model.save_weights('model.h5')
        kf = KFold(KFOLD_SIZE, random_state=42)
        i = 1
        for train_index, test_index in kf.split(X_train):
            train_data, X_test = X_train[train_index], X_train[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]
            model.load_weights('model.h5')
            print(f'Training split {i}')
            current_time = datetime.now()
            model_histories.append(model.fit(train_data, train_labels, validation_data=(X_test, test_labels),
                                             batch_size=BATCH_SIZE, epochs=N_EPOCHS, callbacks=[
                                                 TensorBoard(log_dir=tensorboard_logdir)]
                                             )
                                   )
            training_times.append(datetime.now() - current_time)
            predictions_test = np.argmax(model.predict(X_test), axis=1)
            predictions.append(predictions_test)

            precision_scores.append(precision_score(
                np.argmax(test_labels, axis=1), predictions_test, average='weighted'))
            recall_scores.append(recall_score(
                np.argmax(test_labels, axis=1), predictions_test, average='weighted'))
            f1_scores.append(f1_score(
                np.argmax(test_labels, axis=1), predictions_test, average='weighted'))
            accuracy_scores.append(accuracy_score(
                np.argmax(test_labels, axis=1), predictions_test))
            confusion_matrices.append(confusion_matrix(
                np.argmax(test_labels, axis=1), predictions_test))
            i += 1
    else:
        print('Dataset has a standard train/test split. Using that for test.')
        current_time = datetime.now()
        model_histories.append(model.fit(X_train, labels, validation_data=(X_test, test_labels),
                                         batch_size=BATCH_SIZE, epochs=N_EPOCHS))
        training_times.append(datetime.now() - current_time)
        predictions_test = np.argmax(model.predict(X_test), axis=1)
        predictions.append(predictions_test)

        precision_scores.append(precision_score(
            np.argmax(test_labels, axis=1), predictions_test, average='weighted'))
        recall_scores.append(recall_score(
            np.argmax(test_labels, axis=1), predictions_test, average='weighted'))
        f1_scores.append(f1_score(
            np.argmax(test_labels, axis=1), predictions_test, average='weighted'))
        accuracy_scores.append(accuracy_score(
            np.argmax(test_labels, axis=1), predictions_test))
        confusion_matrices.append(confusion_matrix(
            np.argmax(test_labels, axis=1), predictions_test))

    print('Calculating prediction time...')
    current_time = datetime.now()
    model.predict(X_test)
    prediction_time = datetime.now() - current_time
    avg_prediction_time = prediction_time/len(X_test)

    prediction_file = "{}/{}_{}.predictions".format(
        OUTPUT_DIR_W2V, DATASET_NAME, model_name
    )
    with open(prediction_file, 'wb') as file_predictions:
        pickle.dump(
            predictions, file_predictions
        )

    print('Saving {} model'.format(model_name))
    filename = "{}/{}_{}.pickle".format(
        OUTPUT_DIR_W2V, DATASET_NAME, model_name)
    #os.makedirs(os.path.dirname(OUTPUT_DIR_W2V), exist_ok=True)
    with open(filename, "wb") as file:
        histories = list(map(lambda x: x.history, model_histories))
        pickle.dump([histories, training_times, avg_prediction_time,
                     precision_scores, recall_scores, f1_scores, accuracy_scores, confusion_matrices], file)

print('Done!')
