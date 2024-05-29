import torch
import torch.nn as nn
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from dataset import BilingualDataset, casual_mask
from config import get_weight_file_path, get_config
import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset, DataLoader, random_split

from pathlib import Path
from model import build_transformer
from model import Transformer
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
def greedy_decode(model: Transformer, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    #precompute encoder output and reuse
    
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask)

        #calculate output of decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        prob = model.embed_to_word(out[:,-1]) # word indice

        # val discard, argmax is nextword
        _, next_word = torch.max(prob, dim = 1) # next word rep by the index in tokenizer

        # add new word to decoder
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1,1).type_as(source)
             .fill_(next_word.item()).to(device)], dim = 1
        )    # .item is just make sure it is 1 element
        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)
def run_validation(model: Transformer, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg,
                   global_step, writer, num_examples=2):
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []

    console_width=80
    with torch.no_grad():
        for batch in validation_ds:
            count +=1 
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            assert encoder_input.size(0) == 1, 'batch size is not 1'
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt,
                                      max_len, device)
    
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg('-'*console_width)
            print_msg(f'SOURCE : {source_text}')
            print_msg(f'TARGET : {target_text}')
            print_msg(f'PREDICTED : {model_out_text}')
            
            if count == num_examples:
                break
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()
    

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]
    

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_file :str = config["tokenizer_file"]
    tokenizer_path = Path(tokenizer_file.format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        special_tokens = ['[UNK]', '[PAD]', '[SOS]', '[EOS]']
        trainer = WordLevelTrainer(special_tokens = special_tokens, 
                                   min_frequency= 2,
                                   show_progress = True)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer


def get_ds(config):
    src_lang = config['lang_src']
    tgt_lang = config['lang_tgt']
    seq_len=config["seq_len"]
    ds_raw = load_dataset('opus_books', 
                          f'{src_lang}-{tgt_lang}',
                          split = 'train')
    #ds_raw[0] -> {'id': '0', 'translation': {'en': 'The Wanderer', 'fr': 'Le grand Meaulnes'}}
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, src_lang)
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, tgt_lang)

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, 
                                tokenizer_src=tokenizer_src, 
                                tokenizer_tgt=tokenizer_tgt,
                                src_lang = src_lang,
                                tgt_lang = tgt_lang,
                                seq_len=seq_len)
    val_ds = BilingualDataset(val_ds_raw, 
                                tokenizer_src=tokenizer_src, 
                                tokenizer_tgt=tokenizer_tgt,
                                src_lang = src_lang,
                                tgt_lang = tgt_lang,
                                seq_len=seq_len)
    
    # can also use from sklearn.model_selection import train_test_split

    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][src_lang])
        tgt_ids = tokenizer_tgt.encode(item['translation'][tgt_lang])
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"{max_len_src=}")
    print(f"{max_len_tgt=}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size=config['validation_batch_size'], shuffle = True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len)-> Transformer:
    model = build_transformer(src_token_size = vocab_src_len,
                              tgt_token_size= vocab_tgt_len,
                              src_seq_len= config["seq_len"],
                              tgt_seq_len= config["seq_len"],
                              d_model = config['d_model'], 
                              dropout = 0.1, d_ff= 2048, h =8, N = 6
                              )
    return model
import time
def train_model(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # debug can force cpu to see python stack for small models, device = torch.device('cpu')
    print(f"using {device=}")
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size() ).to(device)

    # tensorboard
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload'] is not None:
        model_filename = get_weight_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        model.load_state_dict(state['model_state_dict'])

        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm.tqdm(train_dataloader, desc = f"Processing epoch {epoch:02d}")
        
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (B, seq len)
            decoder_input = batch['encoder_input'].to(device) # (B, seq len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq len)
            decoder_mask = batch['decoder_mask'].to(device)# (B, 1, seq len, seq len)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            #(B, seq len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, 
                                          decoder_input, decoder_mask)
            #(B, seq len, d_model)
            
            proj_output = model.embed_to_word(decoder_output) # (B, seq len, tgt_vocab_size)
            label: torch.Tensor = batch['label'].to(device) # (B, seq len)
            proj_output: torch.Tensor
            # from typing import List
            # class tokenIDs(int):
            #     pass
            # class label(torch.Tensor, List[tokenIDs]):
            #     pass
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix(dict(loss = f"{loss.item():6.3f}"))

            #log to tensorboard the loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            
            # back prop loss
            loss: nn.CrossEntropyLoss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            

            global_step += 1
            # hardware cooldown protection
            time.sleep(0.2)
            #save model
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, 
                       config['seq_len'], device, 
                       lambda msg: batch_iterator.write(msg), 
                       global_step, writer); 
        model.train()
        model_filename = get_weight_file_path(config, f'{epoch:02d}')
        torch.save(dict(
            epoch = epoch,
            model_state_dict = model.state_dict(),
            optimizer_state_dict = optimizer.state_dict(),
            global_step = global_step
        ), model_filename)


if __name__ == "__main__":
    config = get_config()
    train_model(config)