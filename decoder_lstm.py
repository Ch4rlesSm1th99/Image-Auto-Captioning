import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_mechanisms import CoAttention


class DecoderLSTM(nn.Module):
    def __init__(self, feature_dim, embedding_dim, hidden_dim, vocab_size, d_model, n_head, d_k, d_v, num_layers=1,
                 dropout=0.3):
        super(DecoderLSTM, self).__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # co-attention
        self.co_attention = CoAttention(img_dim=feature_dim, txt_dim=embedding_dim, d_model=d_model, n_head=n_head,
                                        d_k=d_k, d_v=d_v, dropout=dropout)

        # LSTM net
        self.lstm = nn.LSTM(1024, hidden_dim, num_layers, dropout=dropout, batch_first=True)

        # layer for generating vocab
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions):
        # get word embeddings for captions
        embeddings = self.embedding(captions)  # Shape: (batch_size, caption_length, embedding_dim)
        embeddings = self.dropout(embeddings)

        # co-Attention across image nad caption features
        img_attended, txt_attended = self.co_attention(features.unsqueeze(1), embeddings)

        # concat the features along batch dimension, required projecting image features
        img_attended = img_attended.expand(-1, txt_attended.size(1), -1)
        lstm_input = torch.cat((img_attended, txt_attended),
                               dim=2)  # shape: (batch_size, caption_length, feature_dim + embedding_dim)

        # pass through LSTM
        lstm_out, _ = self.lstm(lstm_input)

        # generate vocabulary logits for each time step
        outputs = self.fc(lstm_out)  # Shape: (batch_size, caption_length, vocab_size)

        return outputs

    def generate(self, features, max_length, start_token, end_token, temperature=1.0):
        """
        Generate captions given the image features using greedy or sampling decoding with attention.

        Args:
            features (torch.Tensor): Extracted image features, shape (batch_size, feature_dim).
            max_length (int): Maximum length of the generated caption.
            start_token (int): Index of the <start> token.
            end_token (int): Index of the <end> token.
            temperature (float): Temperature for sampling (default is 1.0).

        Returns:
            torch.Tensor: Generated captions, shape (batch_size, max_length).
        """
        batch_size = features.size(0)
        generated_captions = torch.zeros(batch_size, max_length, dtype=torch.long).to(features.device)
        generated_captions[:, 0] = start_token

        hidden = None
        inputs = self.embedding(generated_captions[:, 0]).unsqueeze(1)  # Shape: (batch_size, 1, embedding_dim)

        for t in range(1, max_length):
            # co-Attention between image features and word embeddings
            img_attended, _ = self.co_attention(features.unsqueeze(1), inputs)

            # concatenate attended image features with word embeddings
            lstm_input = torch.cat((img_attended, inputs), dim=2)  # Shape: (batch_size, 1, feature_dim + embedding_dim)

            # pass through LSTM
            lstm_out, hidden = self.lstm(lstm_input, hidden)  # Shape: (batch_size, 1, hidden_dim)

            # generate vocabulary logits
            logits = self.fc(lstm_out.squeeze(1))  # Shape: (batch_size, vocab_size)

            # greedy decoding
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)  # Sample next token

            generated_captions[:, t] = next_token

            # if all sequences have reached <end> token, stop generating
            if torch.all(next_token == end_token):
                break

            # update inputs for next time step
            inputs = self.embedding(next_token).unsqueeze(1)

        return generated_captions
