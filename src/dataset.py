# 文件: src/dataset.py

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from torchtext.vocab import vocab
import spacy

def get_iwslt_data(batch_size):
    # --- 1. 加载 Spacy 分词器 ---
    try:
        spacy_de = spacy.load('de_core_news_sm')
        spacy_en = spacy.load('en_core_web_sm')
    except IOError:
        print("Downloading spacy models...")
        import os
        os.system("python -m spacy download de_core_news_sm")
        os.system("python -m spacy download en_core_web_sm")
        spacy_de = spacy.load('de_core_news_sm')
        spacy_en = spacy.load('en_core_web_sm')
        
    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]
    
    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    # --- 2. 加载数据集 ---
    dataset = load_dataset("iwslt2017", "iwslt2017-en-de")
    train_data = dataset['train']
    val_data = dataset['validation']

    # --- 3. 构建词典 ---
    def build_vocab(data_iter, tokenizer, language):
        counter = Counter()
        for item in data_iter:
            counter.update(tokenizer(item['translation'][language]))
        specials = ['<unk>', '<pad>', '<bos>', '<eos>']
        v = vocab(counter, specials=specials, min_freq=2)
        v.set_default_index(v['<unk>'])
        return v

    vocab_en = build_vocab(train_data, tokenize_en, 'en')
    vocab_de = build_vocab(train_data, tokenize_de, 'de')

    PAD_IDX = vocab_de['<pad>']
    BOS_IDX = vocab_de['<bos>']
    EOS_IDX = vocab_de['<eos>']

    # --- 4. 数据处理管道 ---
    def data_process(data, lang, v, tokenizer):
        tokens = [v[token] for token in tokenizer(data['translation'][lang])]
        return torch.tensor([BOS_IDX] + tokens + [EOS_IDX])

    def collate_fn(batch):
        src_batch, trg_batch = [], []
        for item in batch:
            src_batch.append(data_process(item, 'en', vocab_en, tokenize_en))
            trg_batch.append(data_process(item, 'de', vocab_de, tokenize_de))
        
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
        trg_batch = pad_sequence(trg_batch, padding_value=PAD_IDX, batch_first=True)
        return src_batch, trg_batch

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, vocab_en, vocab_de, spacy_en, spacy_de