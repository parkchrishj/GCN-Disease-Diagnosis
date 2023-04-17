# GCN-Disease-Diagnosis

## Requirement
Running on Mac OS 11.6.3 (My device)

```shell script
brew install python3
pip3 install pytorch
pip3 install numpy
pip3 install -U scikit-learn scipy matplotlib
```

## data format
```shell script
"filename.nodes.pkl"
# list of node: [node1(str), node2(str), node3(str), ...]

"filename.adj.pkl"
# adj list of nodes: 
# {node1(str): [neighbor1(str), neighbor2(str), ...], node2: []...}

"filename.rare.label.pkl"
# rare flag, indicating whether a node is a rare disease (value=1) 
# or contains a rare disease, NumPy array of shape (N * 1) 

"filename.label.pkl"
# NumPy array of shape (N * D), N is node number 
# and D is the number of diseases

"filename.map.pkl"
# mapping node to index, {node(str): node_id(int), ...}

"filename.train.pkl"
# list of nodes for training, [node_idx_1(int), node_idx_2(int), ....]

"filename.test.pkl"
# list of nodes for testing, [node_idx_1(int), node_idx_2(int), ....]
```

## Run Model
```shell script
python run_multi.py
```
