import pickle
import glob
import numpy as np
from datetime import datetime

OUTPUT_DIR_W2V = 'output_w2v'
EVALUATE_FILE = 'evaluations.txt'
current_time = datetime.now()

eval_file = open(EVALUATE_FILE, 'a')
eval_file.write('>>>>>>------------------------------------')
eval_file.write('Evaluation results for date: {}\n\n'.format(current_time))

eval_file.write('W2V Results:\n')
eval_file.write('------------')
for filename in glob.glob('{}/*.pickle'.format(OUTPUT_DIR_W2V)):
    model_results = pickle.load(open(filename, 'rb'))
    history = model_results[0]
    training_times = model_results[1]
    avg_prediction_time = model_results[2]
    precision_scores = model_results[3]
    recall_scores = model_results[4]
    f1_scores = model_results[5]
    accuracy_scores = model_results[6]
    confusion_matrices = model_results[7]

    val_accs = []
    train_accs = []
    for h in history:
        val_accs.append(h['val_acc'][-1])
        train_accs.append(h['acc'][-1])
    n_epochs = len(h['val_acc'])
    n_folds = len(val_accs)
    mean_val = np.mean(val_accs)
    std_val = np.std(val_accs)
    mean_train = np.mean(train_accs)
    std_train = np.std(train_accs)
    mean_time_per_epoch = np.mean(training_times)/n_epochs
    mean_precision = np.mean(precision_scores)
    std_precision = np.std(precision_scores)
    mean_recall = np.mean(recall_scores)
    std_recall = np.std(recall_scores)
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    mean_conf_matrix = np.mean(confusion_matrices, axis=0)

    eval_file = open(EVALUATE_FILE, 'a')
    eval_file.write('Results for W2V {}:\n'.format(filename))
    eval_file.write(
        'Mean validation accuracy for {} folds: {} ± {}\n'.format(n_folds, mean_val, std_val))
    eval_file.write(
        'Mean training accuracy for {} folds: {} ± {}\n'.format(n_folds, mean_train, std_f1))
    eval_file.write('Mean precision for {} folds: {} ± {}\n'.format(
        n_folds, mean_precision, std_precision))
    eval_file.write(
        'Mean recall for {} folds: {} ± {}\n'.format(n_folds, mean_recall, std_recall))
    eval_file.write(
        'Mean F1 score for {} folds: {} ± {}\n'.format(n_folds, mean_f1, std_f1))
    eval_file.write(
        'Mean Confusion Matrix for {} folds: {}\n'.format(n_folds, mean_conf_matrix))
    eval_file.write(
        'Average training time per epoch: {}\n'.format(mean_time_per_epoch))
    eval_file.write(
        'Average prediction time per sample: {}\n'.format(avg_prediction_time))
    eval_file.write('\n')

eval_file.write('------------\n')
eval_file.write('BERT Results:\n')
eval_file.write('------------')
for filename in glob.glob('output_bert_*/*.pickle'):
    eval_file.write(filename)
    eval_file.write('\n')
    model_results = pickle.load(open(filename, 'rb'))

    eval_file.write('Precision: {}\n'.format(model_results[3]))
    eval_file.write('Recall: {}\n'.format(model_results[4]))
    eval_file.write('F1: {}\n'.format(model_results[5]))
    eval_file.write('Accuracy: {}\n'.format(model_results[6]))
    eval_file.write('Confusion Matrix: {}\n'.format(model_results[7]))
    eval_file.write('Average Prediction Time: {}\n'.format(model_results[2]))

    eval_file.write('Average training time per epoch: {}\n'.format(
        model_results[0]/model_results[1]))
    eval_file.write('\n')

eval_file.write('<<<<<<------------------------------------')
eval_file.flush()
eval_file.close()
