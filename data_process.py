import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

# get data to csv: compound_iso_smiles, target_sequence, label
all_prots = []
dataset = "AS_case"
if dataset == 'AS_case':
    dir_input = ('./data/{}.txt'.format(dataset))
    with open(dir_input, "r") as f:
        train_data_list = f.read().strip().split('\n')
        # train_data_list = shuffle_dataset(train_data_list, seed=34)
drug_id = []
drugs = []
prots = []
labels = []
for pair in train_data_list:
    pair_list = pair.split()
    drug_id.append(pair_list[0])
    lg = Chem.MolToSmiles(Chem.MolFromSmiles(pair_list[2]),isomericSmiles=True)
    drugs.append(lg)
    prots.append(pair_list[1])
    labels.append(pair_list[4])

with open('data/' + dataset + '.csv', 'w') as f:
    f.write('drug_id,compound_iso_smiles,target_sequence,label\n')
    for idx in range(len(labels)):
        ls = []
        ls += [drug_id[idx]]
        ls += [drugs[idx]]
        ls += [prots[idx]]
        ls += [labels[idx]]
        f.write(','.join(map(str,ls)) + '\n')

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

# convert smiles to graph
compound_iso_smiles = []
df = pd.read_csv('data/' + dataset + '.csv')
compound_iso_smiles += list( df['compound_iso_smiles'] )
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g

# get embedding prots vector
prots = np.load('AS_case_prot_embed.npz')
train_proteins = []

processed_data_file = 'data/processed/' + dataset + '.pt'
if (not os.path.isfile(processed_data_file)):
        df = pd.read_csv('data/' + dataset + '.csv')
        train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['label'])
        for item in train_prots:
            prot = prots[item]
            train_proteins.append(prot)
        # XT = [seq_cat(t) for t in train_prots]
        train_drugs, train_prots,  train_Y = np.asarray(train_drugs), np.asarray(train_proteins), np.asarray(train_Y)
        print('preparing ', dataset + '.pt in pytorch format!')
        train_data = TestbedDataset(root='data', dataset=dataset, xd=train_drugs, xt=train_prots, y=train_Y, smile_graph=smile_graph)
else:
    print(processed_data_file,  ' are already created')    
