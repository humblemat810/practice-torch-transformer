import torch
torch.cuda.is_available()

import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """

        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    quick reference, get dropout probit rate: self.dropout.p
    """
    def __init__(self, d_model: int, sql_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.sql_len = sql_len
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        
        pe=torch.zeros(sql_len, d_model)
        position = torch.arange(0, sql_len, dtype= torch.float).unsqueeze(1)
        # the demo used tricks but simply less precise than pure pow
        
        #div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/ d_model))
        to_pow = torch.arange(0, d_model, 2).float() / (-d_model)
        div_term = torch.pow(torch.ones(to_pow.shape[0]) * 10000, to_pow)
        # tested, pow is faster and more accurate in pytorch, see logvspow.py
        
        pe[:,0 :: 2] = torch.sin(position * div_term)
        pe[:,1 :: 2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe_encoding_to_add: torch.Tensor = self.pe[:,:x.shape[1], :]
        x = x + pe_encoding_to_add.requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization (nn.Module):
    def __init__( self, features: int,eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))


    def forward(self, x: torch.Tensor):
        # assume it is [batch .. feat]
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * ((x - mean) / (std + self.eps) ) + self.bias

class FeedForward(nn.Module):
    def __init__(self, 
                 d_model: int, # 512
                 d_ff: int,  # 2048
                 dropout : float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout : nn.Dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiheadAttention(nn.Module):
    def __init__(self, d_model : int,  #512
                 n_head: int = 8, # h in paper
                 dropout: float = 0.1
                 ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        assert d_model % n_head == 0, "d_model has to be multiple of h, num of head"
        self.d_k = d_model // n_head
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    @staticmethod
    def attention(query, key, value, mask = None, dropout = 0.1):
        d_k = query.shape[-1]
        # Batch, head, seq dim, d_v (splitted d_model) * # Batch, head, d_v (splitted d_model)  seq dim
        # i j * j k   -> i k
        attention_scores: torch.Tensor = query @ key.transpose(-2,-1) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)
        # Batch, head, seq dim, seq dim
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        attention_scores: torch.Tensor
        # Batch, head, seq dim, seq dim * Batch, head, seq dim, d_v (splitted d_model)  -> Batch, head, seq dim, d_v (splitted d_model) 
        return attention_scores @ value, attention_scores

    def forward(self, q, k, v, mask):
        # mask same size as beheaded v
        Q: torch.Tensor = self.w_q(q)  # Batch, seq dim, d_model (embedding dim)
        K: torch.Tensor = self.w_k(k) # Batch, seq dim, d_model (embedding dim)
        V: torch.Tensor = self.w_v(v) # Batch, seq dim, d_model (embedding dim)
        Q = Q.view(Q.shape[0], Q.shape[1], self.n_head, self.d_k).transpose(1,2)
        # Batch, head, seq dim, d_v (splitted d_model)
        K = K.view(K.shape[0], K.shape[1], self.n_head, self.d_k).transpose(1,2)
        V = V.view(V.shape[0], V.shape[1], self.n_head, self.d_k).transpose(1,2)

        
        x, self.attention_scores = self.attention(Q, K, V, mask, self.dropout)
        x: torch.Tensor
        # Batch, head, seq dim, d_v  ->Batch,  seq dim, head, d_v (then add and norm later)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.n_head * self.d_k)
        return self.w_o(x)
        
class ResidualConnection(nn.Module):
    def __init__(self, features, dropout = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(features)

    def forward(self, x, sublayer): # delicate pattern
        """We apply dropout [33] to the output of each sub-layer, before it is added to the
sub-layer input and normalized. """
        return self.layer_norm(x + self.dropout(sublayer(x))) # 
        #return x + self.dropout(sublayer(self.layer_norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, features: int, attention_block: MultiheadAttention, feedforward_block: FeedForward, dropout: float):
        super().__init__()
        self.attention_block:MultiheadAttention = attention_block
        self.feedforward_block:FeedForward = feedforward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features = features, dropout=dropout) for _ in range(2)])
    def forward(self, x, mask):
        sublayer = lambda y : self.attention_block(y,y,y, mask)
        x = self.residual_connections[0](x, sublayer)
        sublayer = lambda y : self.feedforward_block(y)
        x = self.residual_connections[1](x, sublayer)
        return x
    
class Encoder(nn.Module):
    """
    
    """
    def __init__(self,features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNormalization(features)
    def forward(self, x, mask):
        for layer in self.layers:
            layer : EncoderBlock
            x = layer(x, mask)
        return self.layer_norm(x)

class DecoderBlock(nn.Module):
    def __init__(self,
                 features, 
                 self_attention : MultiheadAttention,
                 cross_attention: MultiheadAttention,
                 feedforward: FeedForward,
                 dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feedforward = feedforward
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
        
    def forward(self, decoder_x, encoder_x, encoder_mask, mask ):
        decoder_x = self.residual_connections[0](decoder_x, lambda x : self.self_attention(x, x, x ,mask))
        decoder_x = self.residual_connections[1](decoder_x, 
                                                 lambda x : 
                                                        self.cross_attention(x ,encoder_x, encoder_x, encoder_mask))
        decoder_x = self.residual_connections[2](decoder_x, lambda x :self.feedforward(x))

        return decoder_x
    
class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNormalization(features)
    def forward(self, x, encoder_x, encoder_mask, decoder_mask):
        for layer in self.layers:
            layer : DecoderBlock
            x = layer(x,  encoder_x, encoder_mask, decoder_mask)
        return self.layer_norm(x)
    
class EmbeddingsToWord(nn.Module):
    def __init__(self, d_model: int, d_output_word_num: int):
        "dimmension representing the output of word number"
        super().__init__()
        self.linear = nn.Linear(d_model, d_output_word_num)
        
    def forward(self, x):
        #return nn.LogSoftmax(self.linear(x))
        return torch.log_softmax(self.linear(x), dim=-1)

class Transformer(nn.Module):
    
    def __init__(self, encoder: Encoder, decoder: Decoder, 
                 encoder_embeddings: InputEmbeddings, decoder_embeddings: InputEmbeddings,
                 encoder_position_encoder: PositionalEncoding,
                 decoder_position_encoder: PositionalEncoding,
                 embeddings_to_word: EmbeddingsToWord):
        super().__init__()
        self.encoder = encoder 
        self.decoder = decoder
        self.encoder_embeddings = encoder_embeddings
        self.decoder_embeddings = decoder_embeddings
        self.encoder_position_encoder = encoder_position_encoder
        self.decoder_position_encoder = decoder_position_encoder
        self.embeddings_to_word = embeddings_to_word
        
    def encode(self, encoder_x: torch.Tensor, encoder_mask: torch.Tensor):
        encoder_x = self.encoder_embeddings(encoder_x)
        encoder_x = self.encoder_position_encoder(encoder_x)
        return self.encoder(encoder_x, encoder_mask)
    
    def decode(self, encoder_x, encoder_mask, decoder_x, decoder_mask):
        decoder_x = self.decoder_embeddings(decoder_x)
        decoder_x = self.decoder_position_encoder(decoder_x)
        return self.decoder(decoder_x, encoder_x, encoder_mask, decoder_mask)
    
    def embed_to_word(self, x):
        return self.embeddings_to_word(x)
    
def build_transformer(src_token_size: int, tgt_token_size: int, 
                      src_seq_len:int, tgt_seq_len: int,
                      d_model: int = 512, 
                      dropout: float = 0.1,
                      d_ff: int = 2048, # feedforward
                      h: int= 8, # num of head
                      N : int = 6 # num or repeating encoder/ decoder block
                      )-> Transformer: 
    src_embed = InputEmbeddings(d_model, src_token_size)
    tgt_embed = InputEmbeddings(d_model, tgt_token_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout = dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout = dropout)

    encoder_blocks = []

    #seq encoder blocks
    for _ in range(N):
        encoder_self_attention_block = MultiheadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, 
                                     feedforward_block=feed_forward_block, 
                                     dropout=dropout)
        encoder_blocks.append(encoder_block)

    #seq decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiheadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiheadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = DecoderBlock(d_model, decoder_self_attention_block, 
                                     decoder_cross_attention_block,
                                     feedforward=feed_forward_block, 
                                     dropout=dropout)
        decoder_blocks.append(encoder_block)

    #create encoder decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    embeddings_to_word = EmbeddingsToWord(d_model=d_model, 
                                          d_output_word_num= tgt_token_size)
    # create transformer

    transformer = Transformer(encoder = encoder, 
                              decoder = decoder,
                              encoder_embeddings=src_embed,
                              decoder_embeddings=tgt_embed,
                              encoder_position_encoder=src_pos,
                              decoder_position_encoder=tgt_pos,
                              embeddings_to_word= embeddings_to_word
                              )
    
    # initialisation
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer