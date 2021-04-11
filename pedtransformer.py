import torch
import torch.nn as nn
import numpy as np

# BUILD AND CREATE SELF ATTENTION MECHANISM
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        # Length of embedding (length of input vector (object))
        self.embed_size = embed_size
        # The number of heads in the multi head attention
        self.heads = heads
        # The head dimensions will be the embed size divided by the number of heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads ==
                embed_size), "Embed size needs to be divisible by heads"

        # Define linear layers that the keys, values and queries will be sent through
        # torch.nn.Linear(in_features, out_features, bias=True)
        # in_features – size of each input sample
        # out_features – size of each output sample
        # bias – If set to False, the layer will not learn an additive bias. Default: True
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # Each of the above layers is the size of the head size
        # Output layer is size for concatenation of all of the heads (so is embed size again)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        # print(N)
        # The below variables will be different depending when theyre used so define them based on input lengths
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # Energy is the name of the output from multiplying the Queries and the Keys
        energy = torch.einsum("nqhd,nkhd -> nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, keys_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        # Calculate SOFTMAX of the energy output
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        # Multiply the attention output with the values to filter out important values
        out = torch.einsum("nhql,nlhd -> nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # after einsum (N, query_len, heads, head_dim) then flatten last two dimensions

        # Pass through fc_out (dimensions dont change)
        # Just maps the embed size to the embed size
        out = self.fc_out(out)
        return out


# BUILD AND CREATE TRANDFORMER BLOCK
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        # SelfAttention() creates 4 linear nn's; query,keys,values and fc_out
        self.attention = SelfAttention(embed_size, heads)
        # torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True)
        # Applies Layer Normalization over a mini-batch of inputs
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # Define feed forward layers
        # torch.nn.Sequential(*args)
        # A sequential container. 
        self.feed_forward = nn.Sequential(
            
            nn.Linear(embed_size, forward_expansion*embed_size),
            # The rectified linear activation function or ReLU for short is a piecewise
            # linear function that will output the input directly if it is positive,
            # otherwise, it will output zero 
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
    # Define forward pass through transformer block
    def forward(self, value, key, query):
        # Process inputs through attention layers
        attention = self.attention(value, key, query)
        # print('attention output')
        # print(attention)
        # Pass output into first layer norm layer
        # Add the query (residual connection)
        x = self.dropout(self.norm1(attention + query))
        # print('normalisation 1 output:')
        # print(x)
        # Feed through feed forward layers
        forward = self.feed_forward(x)
        # print('feed forward output:')
        # print(forward)
        # Pass output through second layer norm layer
        out = self.dropout(self.norm2(forward + x))
        # print("output from transformer block:")
        # print(out)

        return out

# BUILD AND CREATE BINARY CLASSIFIER
class Classifier(nn.Module):
    def __init__(self, embed_size):
        super(Classifier, self).__init__()
        self.hid1 = nn.Linear(embed_size, 8)  # 4-(8-8)-2
        self.drop1 = nn.Dropout(0.50)
        self.hid2 = nn.Linear(8, 8)
        self.drop2 = nn.Dropout(0.25)
        self.oupt = nn.Linear(8, 2)

        nn.init.xavier_uniform_(self.hid1.weight)
        nn.init.zeros_(self.hid1.bias)
        nn.init.xavier_uniform_(self.hid2.weight)
        nn.init.zeros_(self.hid2.bias)
        nn.init.xavier_uniform_(self.oupt.weight)
        nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = torch.tanh(self.hid1(x)) 
        z = self.drop1(z)
        z = torch.tanh(self.hid2(z))
        z = self.drop2(z)
        z = torch.sigmoid(self.oupt(z))
        return z

# BUILD AND CREATE TRUNK
class Trunk(nn.Module):
    def __init__(
        self,
        embed_size= 12,
        num_layers=3,
        forward_expansion=4,
        heads=2,
        dropout=0.1,
        device="cuda",
        # Maximum length of input sequence
        max_length=1000
    ):
        super(Trunk, self).__init__()
        self.embed_size = embed_size
        self.device = device
        # self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        # 90 corresponds to the number of frames in a 3 second encoding period
        self.position_embedding = nn.Embedding(90, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = Classifier(embed_size)

    def forward(self, x, q, positions):
        # x is the input into the trunk block
        # out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        out = self.dropout(x + self.position_embedding(positions))
        
        q = self.dropout(q)

        # print("Inputs to transformer block:")
        # print("Query:")
        # print(q)
        # print("Clip:")
        # print(out)
        
        for layer in self.layers:
            # This is where we send the inputs into the transformer block
            out = layer(out, out, q)

        # Pass transformer output through classifier
        out = self.classifier(out)
        
        return out

# if __name__ == "__main__":

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Clip representation
#     oldX = torch.tensor([[[0.298, 1.223], [0.645, 0.827], [0.623, 0.885], [0.253, 0.276]]])
#     # Query object in clip
#     oldQ = torch.tensor([[[0.455, 0.892]]])
    
    
#     package = np.load('PIE/testdata/00001.npy', allow_pickle=True)
#     array = package[0]
#     prediction = package[1]
#     length = len(array)
#     query = array[length-1]

#     x = torch.from_numpy(array).unsqueeze(0).float()
#     q = torch.from_numpy(query).unsqueeze(0).unsqueeze(0).float()

#     print(x)
#     print(x.shape)
#     print(q)
#     print(q.shape)


#     # print("The old input array has shape:")
#     # print(oldX)
#     # print(oldX.shape)
#     # print("The new input array has shape:")
#     # print(x)
#     # print(x.shape)
#     # print("The old input query has shape:")
#     # print(oldQ)
#     # print(oldQ.shape)
#     # print("The new input query has shape:")
#     # print(q)
#     # print(q.shape)
    
#     model = Trunk()

#     out = model(x, q)
    
#     print("FINAL OUTPUT:")
#     print(out.shape)
#     print(out)