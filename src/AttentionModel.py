import torch
import torch.nn as nn


class AttentionModel(nn.Module):
    def __init__(self, dimensions):
        super(AttentionModel, self).__init__()
        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights


class MultiheadSelfAttentionModel(nn.Module):
    def __init__(self, dimensions, num_head, max_sentence_length=29):
        super(MultiheadSelfAttentionModel, self).__init__()
        self.num_head = num_head
        self.dimensions = dimensions
        self.max_sentence_length = max_sentence_length
        self.multihead_attentions = nn.ModuleList([AttentionModel(dimensions=dimensions) for _ in range(num_head)])
        # self.pooling = nn.AvgPool1d(max_sentence_length)
        # self.output = nn.Linear(dimensions, 1)

    def forward(self, batch_vectors):
        attention_outputs = []
        for attention_module in self.multihead_attentions:
            outputs, weights = attention_module(query=batch_vectors, context=batch_vectors)
            attention_outputs.append(outputs)
        # pooled_outputs = self.pooling(attention_outputs)

        avg_seq_logits = None
        for l in attention_outputs:
            if avg_seq_logits is None:
                avg_seq_logits = l
            else:
                avg_seq_logits = avg_seq_logits + l
        avg_seq_logits = avg_seq_logits / self.num_head

        # pooled_logits = self.pooling(avg_seq_logits.transpose(2, 1)).transpose(2, 1).squeeze()
        # output = self.output(pooled_logits)
        return avg_seq_logits, None


if __name__ == '__main__':
    a = MultiheadAttentionModel()