import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any
from tokenizers import Tokenizer



class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()

        self.ds = ds
        self.tokenizer_src: Tokenizer = tokenizer_src
        self.tokenizer_tgt: Tokenizer = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id('[SOS]')]).to(torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id('[EOS]')]).to(torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id('[PAD]')]).to(torch.int64)

    def __len__(self):
        return len(self.ds)
        
    def __getitem__(self, index: Any):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        
        enc_input_token = self.tokenizer_src.encode(src_text).ids
        dec_input_token = self.tokenizer_src.encode(tgt_text).ids

        enc_num_padding = self.seq_len - len(enc_input_token) - 2 # SOS and EOS
        dec_num_padding = self.seq_len - len(dec_input_token) - 1 # SOS only, no EOS

        if enc_num_padding < 0 or dec_num_padding < 0:
            raise ValueError("Sentence too long, only support up" 
                             "to seq_len={self.seq_len}"
                             )
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_token, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]* enc_num_padding, dtype = torch.int64)
            ]
        )
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_token, dtype = torch.int64),
                torch.tensor([self.pad_token]* dec_num_padding, dtype = torch.int64)
            ]
        )
                
        label = torch.cat([
                            torch.tensor(dec_input_token, dtype = torch.int64),
                            torch.tensor([self.pad_token]* dec_num_padding, dtype = torch.int64),
                            self.eos_token
                        ])
        return dict(
            encoder_input = encoder_input,
            decoder_input = decoder_input,
            encoder_mask = (encoder_input != self.pad_token).reshape(1,1,self.seq_len).int(),
            label = label,
            decoder_mask = (decoder_input != self.pad_token).reshape(1,1,self.seq_len).int() & casual_mask(decoder_input.size(0)),
            src_text = src_text,
            tgt_text = tgt_text
        )
def casual_mask(size):
    mask = torch.triu(torch.ones(1,size,size), diagonal = 1).type(torch.int)
    return mask == 0

        