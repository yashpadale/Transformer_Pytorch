import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
# Set device to GPU
device = torch.device('cuda:0' )

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(src.device)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(src.device)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=src.device), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
from nltk.tokenize import word_tokenize


def return_order(dict_, content: str):
    # Replace special tokens with placeholders
    content = content.replace('<start>', 'STARTTOKENPLACEHOLDER')
    content = content.replace('<end>', 'ENDTOKENPLACEHOLDER')
    content = content.replace('<space>', 'SPACETOKENPLACEHOLDER')

    # Tokenize the content
    tokens = word_tokenize(content)

    # Replace placeholders back to original tokens
    tokens = ['<start>' if token == 'STARTTOKENPLACEHOLDER' else token for token in tokens]
    tokens = ['<end>' if token == 'ENDTOKENPLACEHOLDER' else token for token in tokens]
    tokens = ['<space>' if token == 'SPACETOKENPLACEHOLDER' else token for token in tokens]

    # Map tokens to their corresponding values in the dictionary
    order = [dict_[token] for token in tokens if token in dict_]

    return order
import  numpy as np
import torch
import torch.nn as nn
import torch.optim as optim




def TrainLanguageModel( src_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout,x_train,y_train,batch_size,epochs):
    transformer=Transformer(
        src_vocab_size, src_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    transformer.train()

    # Create DataLoader for batching
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0
        for src_batch, tgt_batch in train_loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            optimizer.zero_grad()
            output = transformer(src_batch, tgt_batch[:, :-1])

            # Flatten the output and target tensors for loss calculation
            output = output.view(-1, src_vocab_size)
            tgt_batch = tgt_batch[:, 1:].contiguous().view(-1)

            loss = criterion(output, tgt_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch: {epoch + 1}, Loss: {epoch_loss / len(train_loader)}")

    return transformer

def predict(transformer, input_seq, max_seq_length, src_vocab_size):
    device = torch.device("cuda" )
    transformer.eval()

    # Preprocess input sequence
    input_seq = input_seq[:max_seq_length]  # Truncate if needed
    input_seq = input_seq + [0] * (max_seq_length - len(input_seq))  # Pad if needed
    input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = transformer(input_seq, input_seq)

        # Convert logits to probabilities and get predicted token indices
        output = output.view(-1, src_vocab_size)
        predicted_tokens = output.argmax(dim=-1).squeeze().cpu().numpy()

    return predicted_tokens
def process(query,maxlen,dictionary):
    order_=return_order(dictionary,query)
    if len(order_)>maxlen:
        order = query = '<start> ' + query + ' <end> '
        content = pad_segments(content=query, maxlen=maxlen)
        # print("content")
        # print(content)
        order = return_order(dict_=dictionary, content=content)
        # print("order")
        # print(order)
        return order
    else:
        order = return_order(dict_=dictionary, content=query)
        return order

def extract_between_tokens(text, start_token="<start>", end_token="<end>", space_token="<space>"):
    start_pos = text.find(start_token)
    end_pos = text.find(end_token, start_pos)

    # Check if both tokens were found and end_token appears after start_token
    if start_pos == -1 or end_pos == -1 or end_pos <= start_pos:
        return ""

    # Include the end token in the result, so add its length to end_pos
    end_pos += len(end_token)

    # Extract the substring
    substring = text[start_pos:end_pos]

    # Remove the start_token, space_token, and end_token from the substring
    cleaned_text = substring.replace(start_token, "").replace(space_token, "").replace(end_token, "")

    return cleaned_text.strip()

from utils import *
class LanguageModel():
    def __init__(self,src_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        self.src_vocab_size=src_vocab_size
        self.d_model=d_model
        self.num_heads=num_heads
        self.num_layers=num_layers
        self.src_vocab_size=src_vocab_size
        self.d_ff=d_ff
        self.max_seq_length=max_seq_length
        self.dropout=dropout
    def Train(self,x_train,y_train,batch_size,epochs):
        self.transformer=TrainLanguageModel(self.src_vocab_size, self.d_model, self.num_heads, self.num_layers, self.d_ff, self.max_seq_length, self.dropout,x_train,y_train,batch_size,epochs)
        return self.transformer
    def Predict(self,input_seq):
        predicted_sequence = predict(self.transformer, input_seq, self.max_seq_length, self.src_vocab_size)
        return predicted_sequence

    def query_gen_sentences(self,query,dictionary):
        order = process(query, maxlen=self.max_seq_length, dictionary=dictionary)  # converts string to numbers
        output = self.Predict(input_seq=order)
        output = number_to_words(output, dictionary=dictionary)
        return output

    def generate(self,prompt,dictionary):
        output=self.query_gen_sentences(prompt,dictionary=dictionary)
        response=''
        o_=output[0]
        o_=o_.split(' ')
        response+=" "+o_[-1]+" "
        for i in range(len(o_)):
            o=self.query_gen_sentences(output[-1],dictionary=dictionary)
            output.append(o[0])
            o = o[0]
            o= o.split(' ')
            response += " "+o[-1]+" "
        return response

    def generate_text(self,input_prompt,dictionary):
        i = " <start> " + input_prompt + " <end> "
        s1 = pad_segments(i, self.max_seq_length)
        words = self.generate(s1, dictionary=dictionary)
        return words

    def save_model(self, file_path):
        file_path=file_path+'.pt'
        if self.transformer is None:
            raise ValueError("No model to save")
        torch.save(self.transformer, file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        file_path = file_path + '.pt'
        self.transformer = torch.load(file_path)
        print(f"Model loaded from {file_path}")
        return self
