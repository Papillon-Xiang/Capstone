import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

class LSTM_pretrain(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTM_pretrain, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, lengths):
        lengths = lengths.cpu()
        original_len = x.size(1)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, (hn, cn) = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=original_len)

        # reconstruction
        pred = self.output_layer(output)
        # padding prediction
        mask = torch.arange(x.size(1))[None, :] < lengths[:, None]
        mask = mask.to(x.device)
        pred = pred * mask.unsqueeze(-1)
        return pred

    def forward_embedding(self, x, lengths):
        lengths = lengths.cpu()
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, (hn, cn) = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        batch_size = output.size(0)
        all_outputs = []

        for i in range(batch_size):
            patient_length = lengths[i]  # the actual time steps of the ith patient
            patient_output = output[i, :patient_length]
            all_outputs.append(patient_output)

        all_outputs = torch.cat(all_outputs, dim=0)  # [total_timesteps, hidden_dim]

        return all_outputs


class CausalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CausalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query_fc = nn.Linear(hidden_dim, hidden_dim)
        self.key_fc = nn.Linear(hidden_dim, hidden_dim)
        self.value_fc = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()

        # Compute Q, K, V matrices
        Q = self.query_fc(x)  # [batch_size, seq_len, hidden_dim]
        K = self.key_fc(x)  # [batch_size, seq_len, hidden_dim]
        V = self.value_fc(x)  # [batch_size, seq_len, hidden_dim]

        context_vectors = []

        for t in range(seq_len):
            # Only attend to past information
            Q_t = Q[:, t, :].unsqueeze(1)  # [batch_size, 1, hidden_dim]
            K_past = K[:, : t + 1, :]  # [batch_size, t+1, hidden_dim]
            V_past = V[:, : t + 1, :]  # [batch_size, t+1, hidden_dim]

            # Compute attention scores as dot product of Q and K, then apply softmax
            attn_scores = torch.bmm(Q_t, K_past.transpose(1, 2)).squeeze(
                1
            )  # [batch_size, t+1]
            attn_weights = self.softmax(attn_scores)  # [batch_size, t+1]

            # Compute context vector as weighted sum of V_past
            context_vector = torch.bmm(attn_weights.unsqueeze(1), V_past).squeeze(
                1
            )  # [batch_size, hidden_dim]
            context_vectors.append(context_vector)

        return torch.stack(context_vectors, dim=1)  # [batch_size, seq_len, hidden_dim]


class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, attn_layers=1):
        super(LSTMAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.atten_layers = nn.Sequential(
            *[CausalAttention(hidden_dim) for _ in range(attn_layers)]
        )
        self.prediction_head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, lengths):
        lengths = lengths.cpu()
        original_len = x.size(1)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        # LSTM layer
        lstm_out, _ = self.lstm(packed_input)

        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
          lstm_out, 
          batch_first=True,
          total_length=original_len
          )

        # Causal attention layer
        context = self.atten_layers(lstm_out)

        # Prediction head
        output = self.prediction_head(context)

        # padding prediction
        mask = torch.arange(original_len)[None, :] < lengths[:, None]
        mask = mask.to(x.device)
        output = output * mask.unsqueeze(-1)

        return output

    def forward_embedding(self, x, lengths):
        lengths = lengths.cpu()
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        # LSTM layer
        lstm_out, _ = self.lstm(packed_input)

        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        context = self.atten_layers(lstm_out)

        all_outputs = []

        for i in range(context.shape[0]):
            patient_length = lengths[i]  # the actual time steps of the ith patient
            patient_output = context[i, :patient_length]
            all_outputs.append(patient_output)

        all_outputs = torch.cat(all_outputs, dim=0)  # [total_timesteps, hidden_dim]
        return all_outputs


class GPTCausalAttentionModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        num_heads,
        dropout=0.1,
    ):
        super(GPTCausalAttentionModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Define the transformer decoder layer with causal mask
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Prediction head
        self.prediction_head = nn.Linear(hidden_dim, input_dim)

        # Define a positional encoding to encode the sequence order

    def generate_causal_mask(self, seq_len, device):
        # Generates a causal mask with shape (seq_len, seq_len)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        return mask

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Create padding mask to ignore padded tokens (where values are 0)
        padding_mask = x.abs().sum(dim=-1) == 0  # [batch_size, seq_len]

        x = self.embedding(x)

        # Create causal mask for the decoder
        causal_mask = self.generate_causal_mask(seq_len, x.device)

        # Pass through transformer decoder
        transformer_out = self.transformer_decoder(
            x.transpose(0, 1),  # Transformer expects (seq_len, batch, hidden_dim)
            memory=x.transpose(0, 1),
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        ).transpose(
            0, 1
        )  # [batch_size, seq_len, hidden_dim]

        # Prediction head
        output = self.prediction_head(
            transformer_out
        )  # [batch_size, seq_len, output_dim]

        return output

    def forward_embedding(self, x, lengths):
        batch_size, seq_len, _ = x.size()

        # Create padding mask to ignore padded tokens (where values are 0)
        padding_mask = x.abs().sum(dim=-1) == 0  # [batch_size, seq_len]

        x = self.embedding(x)

        # Create causal mask for the decoder
        causal_mask = self.generate_causal_mask(seq_len, x.device)

        # Pass through transformer decoder
        transformer_out = self.transformer_decoder(
            x.transpose(0, 1),  # Transformer expects (seq_len, batch, hidden_dim)
            memory=x.transpose(0, 1),
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        ).transpose(0, 1)

        all_outputs = []

        for i in range(transformer_out.shape[0]):
            patient_length = lengths[i]  # the actual time steps of the ith patient
            patient_output = transformer_out[i, :patient_length]
            all_outputs.append(patient_output)

        all_outputs = torch.cat(all_outputs, dim=0)  # [total_timesteps, hidden_dim]

        return all_outputs


def calculate_class_weights(actions):
    """
    计算5个类别的样本权重
    actions: 包含0-4的标签数组
    return: 每个样本对应的权重数组
    """
    class_counts = np.zeros(5)
    for i in range(5):
        class_counts[i] = np.sum(actions == i)

    eps = 1e-8
    class_counts = np.maximum(class_counts, eps)
    weights = 1.0 / class_counts
    weights = weights / np.sum(weights) * len(weights)
    sample_weights = weights[actions]

    for i in range(5):
        print(f"Class {i}: count = {class_counts[i]}, weight = {weights[i]:.4f}")

    return sample_weights
