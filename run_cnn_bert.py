import tensorflow as tf
import tensorflow_hub as hub
import load_dataset as dataset
import numpy as np
import pickle
import sys
import bert

from bert import run_classifier
from datetime import datetime
from models import model_fn_builder
from tensorflow import set_random_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix

if len(sys.argv) != 3:
    sys.exit('Please provide a dataset name and conf!')

np.random.RandomState(42)
set_random_seed(42)

DATASET_NAME = sys.argv[1]
CONF_TYPE = sys.argv[2]

# Compute train and warmup steps from batch size
# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 10.0
# Warmup is a period of time where the learning rate
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100
IS_TRAINING = True
IS_EVALUATION = True
PRINT_SUMMARY = False

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1"


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        print("get module")
        bert_module = hub.Module(BERT_MODEL_HUB)
        print("tokenization info")
        tokenization_info = bert_module(
            signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            print("sess.run")
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)


# Data

test_data = []
test_labels = []
if DATASET_NAME == 'MR':
    print('Loading MR Dataset')
    X, y = dataset.load_mr()
elif DATASET_NAME == 'TREC':
    print('Loading TREC Dataset')
    X, y, test_data, test_labels = dataset.load_trec()
elif DATASET_NAME == 'PORT_TWITTER':
    print('Loading Portuguese Twitter Dataset')
    X, y = dataset.load_portuguese_twitter()
else:
    sys.exit('DATASET_NAME is not a valid dataset!')

if len(test_data) > 0:
    has_test_data = True
else:
    has_test_data = False

shuffled_indices = list(range(len(X)))
np.random.shuffle(shuffled_indices)

X = X[shuffled_indices]
y = y[shuffled_indices]

label_list = np.unique(y)

print('Defining training and test data...')
# We'll set sequences to be the size of the longest sentence in the dataset.
MAX_SEQ_LENGTH = max(list(map(lambda x: len(x), X)))

if not has_test_data:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42)
else:
    X_train, X_test, y_train, y_test = X, test_data, y, test_labels

# Data Preprocessing
print('Getting train InputExamples...')
train_InputExamples = []
for text, label in zip(X_train, y_train):
    train_InputExamples.append(
        bert.run_classifier.InputExample(guid=None, text_a=text, label=label))

print('Getting test InputExamples...')
test_InputExamples = []
for text, label in zip(X_test, y_test):
    test_InputExamples.append(
        bert.run_classifier.InputExample(guid=None, text_a=text, label=label))

print('Creating tokenizer...')
tokenizer = create_tokenizer_from_hub_module()

print('Converting examples to features...')
# Convert our train and test features to InputFeatures that BERT understands.
train_features = bert.run_classifier.convert_examples_to_features(
    train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
test_features = bert.run_classifier.convert_examples_to_features(
    test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

# Compute number of train steps and warmup steps from batch size
num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

if CONF_TYPE == 'MULTICHANNEL':
    model_names = ['MULTICHANNEL']
    model_confs = [
        {'static': False, 'multichannel': True}
    ]
elif CONF_TYPE == 'NONSTATIC':
    model_names = ['NONSTATIC']
    model_confs = [
        {'static': False, 'multichannel': False},
    ]
elif CONF_TYPE == 'STATIC':
    model_names = ['STATIC']
    model_confs = [
        {'static': True, 'multichannel': False}
    ]
else:
    sys.exit('CONF_TYPE must be STATIC, NONSTATIC or MULTICHANNEL!')

for model_name, model_conf in zip(model_names, model_confs):
    print('Training and evaluating {} model'.format(model_name))

    OUTPUT_DIR = 'output_bert_{}_{}'.format(DATASET_NAME, model_name)
    DO_DELETE = True

    if DO_DELETE:
        try:
            tf.gfile.DeleteRecursively(OUTPUT_DIR)
        except:
            # Doesn't matter if the directory didn't exist
            pass
    tf.gfile.MakeDirs(OUTPUT_DIR)
    print('***** Model output directory: {} *****'.format(OUTPUT_DIR))

    # Specify outpit directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(
        model_dir=OUTPUT_DIR,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
        keep_checkpoint_max=3)

    model_fn = model_fn_builder(
        num_labels=len(label_list),
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        multichannel=model_conf['multichannel'],
        static=model_conf['static'],
        print_summary=PRINT_SUMMARY
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": BATCH_SIZE})

    training_time = None

    if IS_TRAINING:
        # Create an input function for training. drop_remainder = True for using TPUs.
        train_input_fn = bert.run_classifier.input_fn_builder(
            features=train_features,
            seq_length=MAX_SEQ_LENGTH,
            is_training=True,
            drop_remainder=False)

        print('Beginning Training for model {}!'.format(model_name))
        current_time = datetime.now()
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        training_time = datetime.now() - current_time
        print("Training took time ", training_time)

    if IS_EVALUATION:
        test_input_fn = bert.run_classifier.input_fn_builder(
            features=test_features,
            seq_length=MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=False)

        print('Beginning Prediction for model {}!'.format(model_name))
        current_time = datetime.now()
        prediction_results = list(estimator.predict(input_fn=test_input_fn))
        prediction_results = [x['labels'] for x in prediction_results]
        avg_prediction_time = (datetime.now() - current_time)/len(X_test)

        prediction_file = "{}/{}_{}.predictions".format(
            OUTPUT_DIR, DATASET_NAME, model_name
        )
        with open(prediction_file, 'wb') as file_predictions:
            pickle.dump(
                prediction_results, file_predictions
            )

        precision = precision_score(
            y_test, prediction_results, average='weighted')
        recall = recall_score(
            y_test, prediction_results, average='weighted')
        f1_score = f1_score(
            y_test, prediction_results, average='weighted')
        accuracy = accuracy_score(y_test, prediction_results)
        conf_matrix = confusion_matrix(y_test, prediction_results)
        print("Average Prediction time: ", avg_prediction_time)
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1_score))
        print('Accuracy: {}'.format(accuracy))
        print('Confusion Matrix: {}'.format(conf_matrix))

    filename = "{}/{}_{}.pickle".format(
        OUTPUT_DIR, DATASET_NAME, model_name)
    with open(filename, 'wb') as file_out:
        pickle.dump(
            [training_time, NUM_TRAIN_EPOCHS, avg_prediction_time, precision, recall, f1_score, accuracy, conf_matrix], file_out)
