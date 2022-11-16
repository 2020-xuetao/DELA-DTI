import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv,GATConv,SAGEConv, global_max_pool as gmp
import torch.nn.functional as F

class DELADTI(torch.nn.Module):
    def __init__(self, n_output=2, embed_dim=128, num_features_xd=78, num_features_xt=25, hidden_dim=64,
                 num_layer=2, dropout=0.3, bi=True): 
        super(GCNNet, self).__init__()
        self.n_output = n_output 
        # SMILES graph branch
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.conv3 = SAGEConv(num_features_xd*10, num_features_xd*10, 'mean')
        self.fc_xd = torch.nn.Linear(num_features_xd*10, 64*128)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(4, 128)

        #  BiLSTM  protein
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layer, bidirectional=bi)
        self.fc_xt1 = nn.Linear(4, 128)
        
        # attention
        self.attention_layer = nn.Linear(128, 128)
        self.drug_attention_layer = nn.Linear(128, 128)
        self.protein_attention_layer = nn.Linear(128, 128)
 
        self.fcd = nn.Linear(64, 1)
        self.fct = nn.Linear(256, 1)

        # fully connected layer
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein input
        target = data.target  

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)       
        xd = self.fc_xd(x)
        xd = self.relu(xd)
        xd = xd.view(-1, 128, 64)  

        protein = target.view(-1, 256, 4)
        protein = self.fc_xt1(protein)
        protein, _ = self.lstm(protein)  
        xt = protein.permute(0, 2, 1)
        
        drug_att = self.drug_attention_layer(xd.permute(0, 2, 1))
        protein_att = self.protein_attention_layer(xt.permute(0, 2, 1))
        d_att_layers = torch.unsqueeze(drug_att, 2).repeat(1, 1, xt.shape[-1], 1)  
        p_att_layers = torch.unsqueeze(protein_att, 1).repeat(1, xd.shape[-1], 1, 1)  
        Atten_matrix = self.attention_layer(self.relu(d_att_layers + p_att_layers))
        Compound_atte = torch.mean(Atten_matrix, 2) 
        Protein_atte = torch.mean(Atten_matrix, 1)  
        Compound_atte = self.sigmoid(Compound_atte.permute(0, 2, 1)) 
        Protein_atte = self.sigmoid(Protein_atte.permute(0, 2, 1))  

        xd = xd * 0.5 + xd * Compound_atte
        xt = xt * 0.5 + xt * Protein_atte
        xd = self.relu(self.fcd(xd)).squeeze(2)
        xt = self.relu(self.fct(xt)).squeeze(2)

        # concat
        xc = torch.cat((xd, xt), 1) 
        # add some dense layers
        xc = self.dropout(xc)
        xc = self.leaky_relu(self.fc1(xc))
        xc = self.dropout(xc)
        xc = self.leaky_relu(self.fc2(xc))
        xc = self.dropout(xc)
        xc = self.leaky_relu(self.fc3(xc))
        xc = self.dropout(xc)
        predict = self.leaky_relu(self.out(xc))
        return predict
