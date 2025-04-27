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
        '''
        x: input tensor of shape [seq_len, batch_size, input_size]
        h_prev: previous hidden state of shape [1, batch_size, hidden_size]
        '''
        seq_len, batch_size, _ = x.size()
        h_seq = [] 
        h_t = h_prev

        for t in range(seq_len):
            x_t = x[t:t+1, :, :]
            # Concatenate input and previous hidden state
            combined = torch.cat((x_t, h_t), dim=-1)

            # Update gate
            z_t = torch.sigmoid(self.W_zr(combined))
            # Candidate hidden state
            combined_r = torch.cat((x_t, h_t), dim=-1)
            h_candidate = F.relu(self.W_1(combined_r))
            h_candidate = F.relu(self.W_2(h_candidate))
            h_candidate = F.tanh(h_candidate)
            # Updated hidden state
            h_t = (1 - z_t) * h_t + z_t * h_candidate
            h_seq.append(h_t)

        h_seq = torch.stack(h_seq, dim=0).squeeze(1) # shape [seq_len, batch_size, hidden_size]
        
        return h_seq, h_t