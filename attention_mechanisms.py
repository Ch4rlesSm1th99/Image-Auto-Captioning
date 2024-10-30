import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size, len_q, d_model = q.size()

        residual = q

        # seperate the different attention heads
        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k)
        k = self.w_ks(k).view(batch_size, -1, n_head, d_k)
        v = self.w_vs(v).view(batch_size, -1, n_head, d_v)

        # transpose for dot product compatibility
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For n_head dimensions

        q, attn = self.attention(q, k, v, mask=mask)

        # move the head dimension back by transposing
        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class CoAttention(nn.Module):
    def __init__(self, img_dim, txt_dim, d_model, n_head, d_k, d_v, dropout=0.1):
        super(CoAttention, self).__init__()

        # projection layers to map features to an embedding space
        self.img_proj = nn.Linear(2048, d_model)
        self.txt_proj = nn.Linear(txt_dim, d_model)

        # use multi head across modalites
        self.image_to_text_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.text_to_image_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)

    def forward(self, img_features, txt_features):
        # pass through projection layers
        img_features_proj = self.img_proj(img_features)
        txt_features_proj = self.txt_proj(txt_features)

        # apply co attention
        img_attended, _ = self.image_to_text_attention(img_features_proj, txt_features_proj, txt_features_proj)
        txt_attended, _ = self.text_to_image_attention(txt_features_proj, img_features_proj, img_features_proj)

        return img_attended, txt_attended
