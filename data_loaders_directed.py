import torch
import pickle
import os
import ipdb
import numpy as np
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from torch_sparse import coalesce
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx

def load_mail_dataset_direction(path, dataset, coalesce_data=True):
    # then load node labels:
    with open(os.path.join(path, dataset, 'weight_graph.pickle'), 'rb') as f:
        graph = pickle.load(f)
       
    
    graph = np.array(graph)

    hypergraph, copy = {}, {}

    for citation in tqdm(graph):
        ste, t = citation[1], citation[0] # Correct direction of links!
        if t not in hypergraph.keys():
            hypergraph[t], copy[t] = set(), set()
            hypergraph[t].add(t)

        hypergraph[t].add(ste)
        copy[t].add(ste)

    
    # then load node labels:
    with open(os.path.join(path, dataset,  'weight_label.pickle'), 'rb') as f:
        labels = pickle.load(f)
    num_nodes = len(labels)
    print(f'number of nodes:{num_nodes}')
    labels = torch.LongTensor(labels)

    data = processing_without_feature(
        hypergraph=hypergraph,
        labels=labels,
        graph_edge_index=torch.as_tensor(graph, dtype=torch.long).t().contiguous(),
        coalesce_data=coalesce_data,
    )
    return data

# from direcetd graph to undirecetd graph
def load_other_directed_graph(dataset, coalesce_data=True):
    graph = np.array(dataset.edge_index).T

    hypergraph, copy = {}, {}

    for citation in tqdm(graph):
        i, t = citation[1], citation[0] # Correct direction of links!
        if t not in hypergraph.keys():
            hypergraph[t], copy[t] = set(), set()
            hypergraph[t].add(t)

        hypergraph[t].add(i)
        copy[t].add(i)


        
    num_nodes, feature_dim = dataset.x.shape
    assert num_nodes == len(dataset.y)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

   
    data = processing2(hypergraph=hypergraph, data=dataset, coalesce_data=coalesce_data)
    
    return data

def load_synthetic_dataset(path, dataset, coalesce_data=True):
   
    '''
    Dataset loading for syntehtic dataset
    '''

    print(f'Loading Synthetic hypergraph dataset')


    # then load node labels:
    with open(os.path.join(path, dataset, 'label.pickle'), 'rb') as f:
        labels = pickle.load(f)

    num_nodes = len(labels)
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}')

    labels = torch.LongTensor([int(x) for x in labels]) #torch.LongTensor([int(x) -1 for x in labels])

    # The last, load hypergraph.
    with open(os.path.join(path, dataset, 'hypergraph_directed.pickle'), 'rb') as f:
        # hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...}
        hypergraph = pickle.load(f)

    print(f'number of hyperedges: {len(hypergraph)}')


    edge_idx = num_nodes
    node_list = []
    edge_list = []
    num_hyperedges = 0
    edge_weight = []
    # Handling both undirecetd and directed hyperedge!
    for k, v in hypergraph.items():
        if v == [()]:
            cur_he = k
            cur_size = len(cur_he)
            node_list += list(cur_he)
            edge_list += [edge_idx] * cur_size
            edge_idx += 1
            edge_weight += [1] * len(list(cur_he))
            num_hyperedges += 1
        else:
            cur_he1 = list(k)
            cur_he2 = v
            cur_he = cur_he1 + list(cur_he2)
            cur_size = len(cur_he)
            node_list += cur_he
            edge_list += [edge_idx] * cur_size
            edge_idx += 1
            edge_weight += [1j]* len(list(cur_he1)) + [1] * len(list(cur_he2))
            num_hyperedges += 1


# double the weights
    edge_weight = edge_weight * 2 
    edge_index = np.array([node_list + edge_list,
                edge_list + node_list], dtype = np.int64)
    
    edge_index = torch.LongTensor(edge_index)
    edge_weight = torch.tensor(np.array(edge_weight, dtype=np.complex128), dtype=torch.cfloat)
    data = Data(edge_index = edge_index, 
                edge_weight = edge_weight,
                y = labels)
    total_num_node_id_he_id = edge_index.max() + 1
    if coalesce_data:
        data.edge_index, data.edge_weight = coalesce(
            data.edge_index,
            data.edge_weight,
            total_num_node_id_he_id,
            total_num_node_id_he_id,
        )
    data.num_classes = len(np.unique(labels.numpy()))
    data.num_nodes = num_nodes
    data.num_hyperedges = num_hyperedges
    return data    


def load_citation_dataset_direction(path, dataset, coalesce_data=True):
    
    file_name_content = f'{dataset}.content'
    p2idx_features_labels = os.path.join(path, dataset,  file_name_content)
    content = np.genfromtxt(p2idx_features_labels,
                                        dtype=np.dtype(str))
    
    # read citation graph
    file_name = f'{dataset}.cites'
    p2idx_citation_labels = os.path.join(path, dataset, file_name)
    with open(p2idx_citation_labels, "r") as f: 
        cites = f.readlines()
    indices = {j: i for i, j in enumerate(content[:, 0])}
    citations, n = [], 0

    for c in cites:
        c = c.strip("\n").split("\t")
        if c[0] in indices.keys() and c[1] in indices.keys():
            citations.append([c[0], c[1]])
            n = n + 1
    citations = np.array(citations)
    graph = np.array(list(map(indices.get, citations.flatten())), dtype=np.int32).reshape(citations.shape)


    hypergraph, copy = {}, {}

    for citation in tqdm(graph):
        i, t = citation[1], citation[0] # Correct direction of links!
        if t not in hypergraph.keys():
            hypergraph[t], copy[t] = set(), set()
            hypergraph[t].add(t)

        hypergraph[t].add(i)
        copy[t].add(i)




  
        
        # first load node features:
    with open(os.path.join(path, dataset,  'features.pickle'), 'rb') as f:
        features = pickle.load(f)
        features = features.todense()

    # then load node labels:
    with open(os.path.join(path, dataset,  'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
 

    data = processing(
        hypergraph=hypergraph,
        labels=labels,
        features=features,
        graph_edge_index=torch.as_tensor(graph, dtype=torch.long).t().contiguous(),
        coalesce_data=coalesce_data,
    )
    return data

#Dataset loading for chameleon and squirrel dataset
def _npz_edges_to_edge_index(edges_np: np.ndarray) -> torch.Tensor:

    if not isinstance(edges_np, np.ndarray):
        raise TypeError("`edges` must be a NumPy array.")
    if edges_np.dtype == object:
        raise ValueError("Object-dtype `edges` not supported here. Provide a (2,E) or (E,2) int array.")

    if edges_np.ndim != 2:
        raise ValueError("`edges` must be 2D, got shape {}.".format(edges_np.shape))

    # If edges are (E,2), transpose to (2,E)
    if edges_np.shape[0] != 2 and edges_np.shape[1] == 2:
        edges_np = edges_np.T
    elif edges_np.shape[0] == 2:
        pass
    else:
        raise ValueError("`edges` must be of shape (2,E) or (E,2); got {}.".format(edges_np.shape))

    return torch.as_tensor(edges_np, dtype=torch.long)

def load_additional_dataset(path: str, dataset: str, coalesce_data=True) -> Data:

    npz_file = os.path.join(path, dataset, f"{dataset}_filtered.npz")
    npz = np.load(npz_file, allow_pickle=True)

    x_np = npz["node_features"]
    y_np = npz["node_labels"]

    x = torch.as_tensor(x_np, dtype=torch.float32)
    y = torch.as_tensor(y_np, dtype=torch.long)

    edge_index = _npz_edges_to_edge_index(npz["edges"])

    def _maybe_mask(key):
        if key in npz.files:
            t = torch.as_tensor(npz[key])
            return t if t.dtype == torch.bool else (t > 0)
        return None

    train_mask = _maybe_mask("train_masks").T
    val_mask   = _maybe_mask("val_masks").T
    test_mask  = _maybe_mask("test_masks").T

    dataset = Data(x=x, y=y, edge_index=edge_index)

    if train_mask is not None: dataset.train_mask = train_mask
    if val_mask   is not None: dataset.val_mask   = val_mask
    if test_mask  is not None: dataset.test_mask  = test_mask

    # graph: E x 2 (src, dst)
    graph = edge_index.T.cpu().numpy()

    hypergraph, copy = {}, {}

    # For each directed edge (src -> dst), create/extend a hyperedge anchored at src (t)
    # and containing itself plus its out-neighbor (dst).
    for citation in graph:
        t = int(citation[0])  # source (tail)
        i = int(citation[1])  # target (head)

        if t not in hypergraph:
            hypergraph[t], copy[t] = set(), set()
            hypergraph[t].add(t)  # include the tail itself

        hypergraph[t].add(i)
        copy[t].add(i)

    num_nodes, feature_dim = dataset.x.shape
    assert num_nodes == dataset.y.numel()
    print(f"number of nodes:{num_nodes}, feature dimension: {feature_dim}")

    data = processing2(hypergraph=hypergraph, data=dataset, coalesce_data=coalesce_data)
    return data

def processing(
    hypergraph,
    labels,
    features,
    graph_edge_index=None,
    coalesce_data=True,
):
        num_nodes = len(labels)
        edge_idx = num_nodes
        node_list = []
        edge_list = []
        edge_weight = []
        for k, v in hypergraph.items():
            cur_he1 = [k]
            cur_he2 = [node for node in v if node != k]
            cur_he = list(cur_he1) + list(cur_he2)
            cur_size = len(cur_he)
            node_list += list(cur_he)
            edge_list += [edge_idx] * cur_size
            edge_idx += 1
            edge_weight += [1j]* len(list(cur_he1)) + [1] * len(list(cur_he2))

        # double the weights
        edge_weight = edge_weight * 2 

        edge_index = np.array([node_list + edge_list,
                    edge_list + node_list], dtype = np.int64)
        

        edge_index = torch.LongTensor(edge_index)
        edge_weight = torch.tensor(np.array(edge_weight, dtype=np.complex128), dtype=torch.cfloat)
        data = Data(x = features,
                    edge_index = edge_index, 
                    edge_weight = edge_weight,
                    y = labels)
        if graph_edge_index is not None:
            data.graph_edge_index = graph_edge_index.long()

        total_num_node_id_he_id = edge_index.max() + 1
        if coalesce_data:
            data.edge_index, data.edge_weight = coalesce(
                data.edge_index,
                data.edge_weight,
                total_num_node_id_he_id,
                total_num_node_id_he_id,
            )
        data.num_classes = len(np.unique(labels.numpy()))
        data.num_nodes = num_nodes
        data.num_hyperedges = len(hypergraph)
        return data

def processing_without_feature(
    hypergraph,
    labels,
    graph_edge_index=None,
    coalesce_data=True,
):
        num_nodes = len(labels)
        edge_idx = num_nodes
        node_list = []
        edge_list = []
        edge_weight = []
        for k, v in hypergraph.items():
            cur_he1 = [k]
            cur_he2 = [node for node in v if node != k]
            cur_he = list(cur_he1) + list(cur_he2)
            cur_size = len(cur_he)
            node_list += list(cur_he)
            edge_list += [edge_idx] * cur_size
            edge_idx += 1
            edge_weight += [1j]* len(list(cur_he1)) + [1] * len(list(cur_he2))

        # double the weights
        edge_weight = edge_weight * 2 

        edge_index = np.array([node_list + edge_list,
                    edge_list + node_list], dtype = np.int64)
        

        edge_index = torch.LongTensor(edge_index)
        edge_weight = torch.tensor(np.array(edge_weight, dtype=np.complex128), dtype=torch.cfloat)
        data = Data(edge_index = edge_index, 
                    edge_weight = edge_weight,
                    y = labels)
        if graph_edge_index is not None:
            data.graph_edge_index = graph_edge_index.long()

        total_num_node_id_he_id = edge_index.max() + 1
        if coalesce_data:
            data.edge_index, data.edge_weight = coalesce(
                data.edge_index,
                data.edge_weight,
                total_num_node_id_he_id,
                total_num_node_id_he_id,
            )
        data.num_classes = len(np.unique(labels.numpy()))
        data.num_nodes = num_nodes
        data.num_hyperedges = len(hypergraph)
        return data

def processing2(hypergraph, data, coalesce_data=True):
        n =  len(data.y)
        d = len(hypergraph)
        data.graph_edge_index = data.edge_index.clone().long()
    
        # Initialize a matrix of zeros
        #H = np.zeros((n, d), dtype=np.complex128)
        values = []
        rows = []
        cols = []
        # Populate the matrix based on the dictionary
        print('sono pronto a creare il grafo')
        for i, (key, subdict) in enumerate(hypergraph.items()):
        
            rows.append(key)
            cols.append(i)
            values.append(1j)
            for raw in subdict:
                if raw!= key:
                    rows.append(raw)
                    cols.append(i)
                    values.append(1)
    

        H = torch.sparse_coo_tensor(torch.tensor([rows, cols]), torch.tensor(values), torch.Size([n, d])).coalesce()
      
        
        edge_index = torch.LongTensor(H.indices())
        edge_weight = torch.tensor(np.array(H.values(), dtype=np.complex128), dtype=torch.cfloat)

        data.edge_index = edge_index
        data.edge_weight = edge_weight
        total_num_node_id_he_id = edge_index.max() + 1
        if coalesce_data:
            data.edge_index, data.edge_weight = coalesce(
                data.edge_index,
                data.edge_weight,
                total_num_node_id_he_id,
                total_num_node_id_he_id,
            )
        data.num_classes = len(np.unique(data.y.numpy()))
        data.num_nodes = n
        data.num_hyperedges = len(hypergraph)
        return data

def load_citation_pubmed_dataset_direction_reverse(path, dataset, coalesce_data=True):
    
    # then load node labels:
    with open(os.path.join(path, dataset,  'hypergraph.pickle'), 'rb') as f:
        hypergraph = pickle.load(f)
    # Create an array of indices for each element in hypergraph2
    reverse_graph = []

    for s, neighbors in hypergraph.items():
        for c in neighbors:
            reverse_graph.append([c, s])
    
    
    graph = np.array(reverse_graph)#- 1
    #graph = np.array(list(map(node_map.get, citations.flatten())), dtype=np.int32).reshape(citations.shape)
    hypergraph, copy = {}, {}

    for citation in tqdm(graph):
        ste, t = citation[1], citation[0] # Correct direction of links!
        if t not in hypergraph.keys():
            hypergraph[t], copy[t] = set(), set()
            hypergraph[t].add(t)

        hypergraph[t].add(ste)
        copy[t].add(ste)


    
    #    # first load node features:
    with open(os.path.join(path, dataset,  'features.pickle'), 'rb') as f:
        features = pickle.load(f)
        features = features.todense()

    # then load node labels:
    with open(os.path.join(path, dataset,  'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    data = processing(
        hypergraph=hypergraph,
        labels=labels,
        features=features,
        graph_edge_index=torch.as_tensor(graph, dtype=torch.long).t().contiguous(),
        coalesce_data=coalesce_data,
    )
    return data

def load_citation_pubmed_dataset_direction(path, dataset, coalesce_data=True):
    
    file_name_content = 'Pubmed-Diabetes.NODE.paper.tab'
    p2idx_features_labels = os.path.join(path, dataset,  file_name_content)
    num_feats = 500
    num_nodes = 19717 
    feat_data = np.zeros((num_nodes, num_feats))
    labels_2 = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open(p2idx_features_labels) as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]: i - 1 for i, entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels_2[i] = int(info[1].split("=")[1]) - 1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])



     # read citation graph
    file_name = 'Pubmed-Diabetes.DIRECTED.cites.tab'
    p2idx_citation_labels = os.path.join(path, dataset, file_name)
    citations, n = [], 0
    with open(p2idx_citation_labels, "r") as f: 
        cites = f.readlines()
    for c in cites:
        try:
            c = c.strip().split("\t")
            paper1 = node_map[c[1].split(":")[1]]
            paper2 = node_map[c[-1].split(":")[1]]
            citations.append([paper1, paper2])
            n = n + 1
        except:
            continue
    graph = np.array(citations)
    
    hypergraph, copy = {}, {}

    for citation in tqdm(graph):
        ste, t = citation[1], citation[0] # Correct direction of links!
        if t not in hypergraph.keys():
            hypergraph[t], copy[t] = set(), set()
            hypergraph[t].add(t)

        hypergraph[t].add(ste)
        copy[t].add(ste)
    


    # first load node features:
    with open(os.path.join(path, dataset, 'features.pickle'), 'rb') as f:
        features = pickle.load(f)
        features = features.todense()
    # then load node labels:
    with open(os.path.join(path, dataset, 'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)
    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')
    features = torch.FloatTensor(features)
    #labels = torch.LongTensor(labels)
    labels = torch.LongTensor(labels_2.flatten())
    
    data = processing(
        hypergraph=hypergraph,
        labels=labels,
        features=features,
        graph_edge_index=torch.as_tensor(graph, dtype=torch.long).t().contiguous(),
        coalesce_data=coalesce_data,
    )
    return data


