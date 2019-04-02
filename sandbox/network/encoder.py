"""
Vanilla RNN encoder, supports RNN|LSTM
"""
from __future__ import division

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from sandbox.utils.misc import aeq
from torch.autograd import Variable
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence

class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :obj:`sandbox.network.models.NMTModel`.

    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
            C[Pos 1]
            D[Pos 2]
            E[Pos N]
          end
          F[Memory_Bank]
          G[Final]
          A-->C
          A-->D
          A-->E
          C-->F
          D-->F
          E-->F
          E-->G
    """

    def _check_args(self, src, lengths=None, hidden=None):
        _, n_batch, _ = src.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, src, lengths=None):
        """
        Args:
            src (:obj:`LongTensor`):
               padded sequences of sparse indices `[src_len x batch x nfeat]`
            lengths (:obj:`LongTensor`): length of each sequence `[batch]`


        Returns:
            (tuple of :obj:`FloatTensor`, :obj:`FloatTensor`):
                * final encoder state, used to initialize decoder
                * memory bank for attention, `[src_len x batch x hidden]`
        """
        raise NotImplementedError

class RNNLayer(EncoderBase):
    """
        custom RNN layer that merges output if O_t + O_(t-1) > packet length
        #TODO add vocab lookup and token merge
        #TODO Fix batching, iterate individual each ones
        #pack sequence(not necessary...)
    """
    def __init__(self, rnn_type, hidden_size, n_layers, dropout, embedding=None):
        super(RNNLayer, self).__init__()

        assert embedding != None
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.input_size = self.embedding.embedding_size
        self.rnnCell = nn.RNNCell(self.input_size, self.hidden_size)

    def forward(self, src, lengths=None):

        embedded = self.embedding(src)
        assert embedded.size(0) == lengths[0]                           # sanity check
        time_steps = embedded.size(0)                                   # get time step from embed
        outputs = []
        hidden = None
        for t in range(time_steps):
            src_input = embedded[t]  
            hidden = self.rnnCell(src_input, hidden)
            outputs.append(hidden)

        layer_final =  hidden 
        outputs = torch.stack(outputs)
        print("layer Model's state_dict:")
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())
        return layer_final, outputs                                             

class RNNEncoder(EncoderBase):
    ''' 
        rnn encoder with extra layer
    '''
    def __init__(self, rnn_type, hidden_size, n_layers, dropout, embedding=None):
        super().__init__()
        self.no_pack_padded_seq = False

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        assert embedding != None
        self.embedding = embedding
        
        # add extra rnn layer
        self.rnnLayer = RNNLayer(rnn_type, hidden_size, n_layers, dropout, embedding)
        self.rnn = getattr(nn, rnn_type)(self.embedding.embedding_size, hidden_size, n_layers, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, lengths=None):

        self._check_args(src, lengths)
        #src = [src sent len, batch size]

        layer_final, outputs = self.rnnLayer(src, lengths)
        # print(layer_final.size())
        # print(outputs.size())
        layer_in = layer_final.expand(self.n_layers, *layer_final.size()).contiguous()
        # print("new layer size is ", layer_in.size())
        #embedded = [src sent len, batch size, emb dim]
        cell_n = layer_in.new_zeros(*layer_in.size(), requires_grad=False).contiguous()

        memory_bank, encoder_final = self.rnn(outputs, (layer_in, cell_n))

        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        # print(memory_bank)
        # print(encoder_final)

        print("encoder Model's state_dict:")
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())

        return encoder_final, memory_bank



class vanillaRNNEncoder(EncoderBase):
    ''' 
        vanilla rnn encoder, supports RNN|lstm
    '''
    def __init__(self, rnn_type, hidden_size, n_layers, dropout, embedding=None):
        super().__init__()
        self.no_pack_padded_seq = False

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        assert embedding != None
        self.embedding = embedding
        
        self.rnn = getattr(nn, rnn_type)(self.embedding.embedding_size, hidden_size, n_layers, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, lengths=None):

        self._check_args(src, lengths)
        #src = [src sent len, batch size]
        embedded = self.embedding(src)
        packed_emb = embedded

        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(embedded, lengths)

        #embedded = [src sent len, batch size, emb dim]
        memory_bank, encoder_final = self.rnn(packed_emb)
        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        return encoder_final, memory_bank
