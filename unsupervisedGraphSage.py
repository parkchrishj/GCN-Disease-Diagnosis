# brew install python3
# pip3 install pytorch
# pip3 install numpy
# pip3 install -U scikit-learn scipy matplotlib

import numpy as np
from Model.unsupervised import RarePredictor
import time
import pickle


file_path = "./data/sample_data/sample_garph"
node_list = pickle.load(open(file_path + ".nodes.pkl", "rb"))
adj_lists = pickle.load(open(file_path + ".adj.pkl", "rb"))
rare_patient = pickle.load(open(file_path + ".rare.label.pkl", "rb"))
labels = pickle.load(open(file_path + ".label.pkl", "rb"))
node_map = pickle.load(open(file_path + ".map.pkl", "rb"))
train = pickle.load(open(file_path + ".train.pkl", "rb"))
test = pickle.load(open(file_path + ".test.pkl", "rb"))

multi_class_num = 108
feature_dim = 10000
epoch = 8000
batch_num = 200
lr = 0.3
feat_data = np.random.random((50000, feature_dim))
train_enc_dim = [1000, 1000, 1000, 1000]
t1 = time.time()
model = RarePredictor(feat_data=feat_data,
                      b_labels=rare_patient,
                      multi_class_num=108,
                      m_labels=labels,
                      adj_lists=adj_lists,
                      feature_dim=feature_dim,
                      train_enc_num=1,
                      train_enc_dim=train_enc_dim,
                      train_sample_num=[5, 5, 5, 5],
                      train=train, test=test,
                      pos_samp_num=1, neg_samp_num=1,
                      attention=False, weights_flag=False, weights=[0.5, 0.5],
                      cuda=False,
                      N_WALK_LEN=5, N_WALKS=6)

model.run_unsupervised(loop_num=100, batch_num=512,
                       lr=0.01)  # epoch, batch_num, lr
print(feature_dim, train_enc_dim)
print("running time:", time.time()-t1, "s")
