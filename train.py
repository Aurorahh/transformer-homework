# 文件: train.py 

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import os
import math
from tqdm import tqdm

from src.model import Transformer
from src.dataset import get_iwslt_data
# 导入 set_seed
from src.utils import count_parameters, plot_curves, calculate_bleu, translate_sentence, set_seed

# --- 超参数 ---
BATCH_SIZE = 64 
EMBED_DIM = 256
# N_HEADS 和 N_LAYERS 将从命令行读取
FF_DIM = 512
DROPOUT = 0.1
LEARNING_RATE = 0.0005
EPOCHS = 30
CLIP = 1.0


def train(model, loader, optimizer, criterion, clip, device, src_pad_idx, trg_pad_idx):
    model.train()
    epoch_loss = 0
    for batch in tqdm(loader, desc="Training"):
        src, trg = batch
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        output = model(src, trg[:,:-1], src_pad_idx, trg_pad_idx)
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(loader)


def evaluate(model, loader, criterion, device, src_pad_idx, trg_pad_idx):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            src, trg = batch
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg[:,:-1], src_pad_idx, trg_pad_idx)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer Seq2Seq Training')
    parser.add_argument('--no-pos-encoding', action='store_true', help='Disable positional encoding')
    parser.add_argument('--experiment-name', type=str, default='baseline_mt', help='Name for experiment')
    
    # --- 新增的命令行参数 ---
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--n-layers', type=int, default=3, help='Number of encoder/decoder layers')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    
    args = parser.parse_args()
    
    # --- 在一切开始前设置随机种子 ---
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 数据加载 ---
    train_loader, val_loader, vocab_en, vocab_de, spacy_en, spacy_de = get_iwslt_data(BATCH_SIZE)
    SRC_VOCAB_SIZE = len(vocab_en)
    TRG_VOCAB_SIZE = len(vocab_de)
    SRC_PAD_IDX = vocab_en['<pad>']
    TRG_PAD_IDX = vocab_de['<pad>']

    # --- 模型初始化 ---
    use_pos = not args.no_pos_encoding
    print(f"Positional Encoding Enabled: {use_pos}")
    print(f"Hyperparameters: N_LAYERS={args.n_layers}, N_HEADS={args.n_heads}, SEED={args.seed}")
    
    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        trg_vocab_size=TRG_VOCAB_SIZE,
        d_model=EMBED_DIM,
        n_layers=args.n_layers,   # <-- 使用 args
        n_heads=args.n_heads,     # <-- 使用 args
        d_ff=FF_DIM,
        dropout=DROPOUT,
        use_pos_encoding=use_pos
    ).to(device)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, CLIP, device, SRC_PAD_IDX, TRG_PAD_IDX)
        valid_loss = evaluate(model, val_loader, criterion, device, SRC_PAD_IDX, TRG_PAD_IDX)
        end_time = time.time()
        
        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), f'checkpoints/{args.experiment_name}_best_model.pt')
            
        print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs:.0f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    plot_curves(train_losses, val_losses, args.experiment_name)
    
    # --- 最终评估与定性分析 ---
    model.load_state_dict(torch.load(f'checkpoints/{args.experiment_name}_best_model.pt'))
    bleu = calculate_bleu(val_loader, model, vocab_en, vocab_de, spacy_de, device)
    print(f"Final BLEU score on validation set: {bleu*100:.2f}")

    # --- 新增：保存结果到表格 ---
    if not os.path.exists('results/tables'):
        os.makedirs('results/tables')

    summary_file = "results/tables/experiment_summary.md"
    
    # 检查文件是否为空，如果为空则写入表头
    if not os.path.exists(summary_file) or os.path.getsize(summary_file) == 0:
        with open(summary_file, 'w') as f:
            f.write("| Experiment Name | N Layers | N Heads | Best Val Loss | Best Val PPL | BLEU Score |\n")
            f.write("|---|---|---|---|---|---|\n")

    # 写入当前实验结果
    with open(summary_file, 'a') as f:
        best_ppl = math.exp(best_val_loss)
        f.write(f"| {args.experiment_name} | {args.n_layers} | {args.n_heads} | {best_val_loss:.3f} | {best_ppl:7.3f} | {bleu*100:.2f} |\n")

    print(f"Results saved to {summary_file}")
    # --- 结束 ---

    test_sentence = "a man in a blue shirt is running on the beach."
    translation = translate_sentence(test_sentence, model, vocab_en, vocab_de, spacy_en, device)
    print(f"\n--- Qualitative Example ---")
    print(f"Source: {test_sentence}")
    print(f"Translated: {translation}")