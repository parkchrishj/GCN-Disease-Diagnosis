# brew install python3
# pip3 install pytorch
# pip3 install numpy
# pip3 install -U scikit-learn scipy matplotlib

import numpy as np
from Model.model_multi_diff_weights import DiseasesPredictor
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
epoch = 80
batch_num = 10
lr = 0.3
feat_data = np.random.random((50000, feature_dim))
train_enc_dim = [1000, 1000, 1000, 1000]
t1 = time.time()
model = DiseasesPredictor(feat_data=feat_data,
                          b_labels=rare_patient,
                          multi_class_num=108,
                          labels=labels,
                          adj_lists=adj_lists,
                          feature_dim=feature_dim,
                          train_enc_num=1,
                          train_enc_dim=train_enc_dim,
                          train_sample_num=[5, 5, 5, 5],
                          train=train, test=test,
                          kernel='GCN',
                          topk=(1, 2, 3, 4, 5,))

model.run(epoch, batch_num, lr)  # epoch, batch_num, lr
print(feature_dim, train_enc_dim)
print("running time:", time.time()-t1, "s")
