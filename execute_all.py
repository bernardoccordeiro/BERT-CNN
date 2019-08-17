import os

print('Running W2V with MR')
os.system(r"python3 run_cnn_w2v.py MR")

print('Running W2V with TREC')
os.system(r"python3 run_cnn_w2v.py TREC")

print('Running W2V with Portuguese Twitter')
os.system(r"python3 run_cnn_w2v.py PORT_TWITTER")

print('Running BERT with MR, Static Conf')
os.system(r"python3 run_cnn_bert.py MR STATIC")

print('Running BERT with MR, Nonstatic Conf')
os.system(r"python3 run_cnn_bert.py MR NONSTATIC")

print('Running BERT with TREC')
os.system(r"python3 run_cnn_bert.py TREC STATIC")

print('Running BERT with TREC')
os.system(r"python3 run_cnn_bert.py TREC NONSTATIC")

print('Running BERT with Portuguese Twitter')
os.system(r"python3 run_cnn_bert.py PORT_TWITTER STATIC")

print('Running BERT with Portuguese Twitter')
os.system(r"python3 run_cnn_bert.py PORT_TWITTER NONSTATIC")

print('Running BERT with MR')
os.system(r"python3 run_cnn_bert.py MR MULTICHANNEL")

print('Running BERT with TREC')
os.system(r"python3 run_cnn_bert.py TREC MULTICHANNEL")

print('Running BERT with Portuguese Twitter')
os.system(r"python3 run_cnn_bert.py PORT_TWITTER MULTICHANNEL")
