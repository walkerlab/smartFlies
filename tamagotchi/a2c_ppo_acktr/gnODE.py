import torch
import torch.nn as nn
import torch.nn.functional as F

class GNODE(nn.Module):
    def __init__(self, input_size, hidden_size, layer_size):
        super(GNODE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size

        # Weight matrices for update gate
        self.W_zr = nn.Linear(input_size + hidden_size, hidden_size)
        # Weight matrices for candidate hidden state
        self.W_1 = nn.Linear(input_size + hidden_size, layer_size)
        # Weight matrices for candidate hidden state
        self.W_2 = nn.Linear(layer_size, hidden_size)

    def forward(self, x, h_prev):
        h_t = h_prev
        combined = torch.cat((x, h_t), dim=2)
        # Update gate
        z_t = torch.sigmoid(self.W_zr(combined))
        # Candidate hidden state
        combined_r = torch.cat((x, h_t), dim=2)
        h_candidate = F.relu(self.W_1(combined_r))
        h_candidate = F.relu(self.W_2(h_candidate))
        h_candidate = F.tanh(h_candidate)
        # Updated hidden state
        h_t = (1 - z_t) * h_t + z_t * h_candidate
        
        return h_t, h_t # match output structure of VRNN