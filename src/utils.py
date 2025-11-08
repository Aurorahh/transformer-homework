# 文件: src/utils.py

import torch
import matplotlib.pyplot as plt
import os
from torchtext.data.metrics import bleu_score
import random
import numpy as np

def set_seed(seed):
    """
    设置随机种子以确保可复现性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 确保CUDNN的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_curves(train_losses, val_losses, experiment_name):
    if not os.path.exists('results/training_curves'):
        os.makedirs('results/training_curves')
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Loss Curves for {experiment_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/training_curves/{experiment_name}_loss_curve.png')
    plt.close()
    print(f"Loss curve saved.")

#
# --- 关键修复：在这里添加了 vocab_en ---
#
def calculate_bleu(data_loader, model, vocab_en, vocab_de, spacy_de, device):
    trgs = []
    pred_trgs = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            src, trg = batch
            src, trg = src.to(device), trg.to(device)
            
            # --- 现在 vocab_en 是已定义的, 这行代码可以工作了 ---
            src_pad_idx = vocab_en['<pad>'] 
            trg_pad_idx = vocab_de['<pad>']
            
            output = model(src, trg[:, :-1], src_pad_idx, trg_pad_idx) # exclude <eos>
            
            pred_tokens = output.argmax(2).detach().cpu().numpy().tolist()
            
            for i in range(len(pred_tokens)):
                pred = [vocab_de.get_itos()[tok] for tok in pred_tokens[i]]
                target = [vocab_de.get_itos()[tok] for tok in trg[i, 1:].cpu().numpy()] # exclude <bos>
                
                # 去掉<eos>之后的部分
                try:
                    eos_index = pred.index('<eos>')
                    pred = pred[:eos_index]
                except ValueError:
                    pass
                try:
                    eos_index = target.index('<eos>')
                    target = target[:eos_index]
                except ValueError:
                    pass

                pred_trgs.append(pred)
                trgs.append([target]) # bleu_score expects a list of references

    try:
        # 明确传入 4-gram 的权重，这可以防止 torchtext 的内部索引错误
        return bleu_score(pred_trgs, trgs, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])
    except IndexError:
        print("WARNING: BLEU score calculation failed (likely due to empty predictions). Returning 0.0")
        return 0.0 # 在最坏情况下返回 0
    except Exception as e:
        print(f"An unexpected error occurred during BLEU calculation: {e}. Returning 0.0")
        return 0.0

def translate_sentence(sentence, model, vocab_en, vocab_de, spacy_en, device, max_len=50):
    model.eval()
    tokens = [tok.text.lower() for tok in spacy_en.tokenizer(sentence)]
    tokens = ['<bos>'] + tokens + ['<eos>']
    src_indexes = [vocab_en.get_stoi().get(token, vocab_en['<unk>']) for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_pad_idx = vocab_en['<pad>']
        
    src_mask = model.make_src_mask(src_tensor, src_pad_idx)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [vocab_de['<bos>']]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        
        trg_pad_idx = vocab_de['<pad>']
            
        trg_mask = model.make_trg_mask(trg_tensor, trg_pad_idx)
        with torch.no_grad():
            output = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(pred_token)
        if pred_token == vocab_de['<eos>']:
            break
    
    trg_tokens = [vocab_de.get_itos()[i] for i in trg_indexes]
    return " ".join(trg_tokens[1:-1]) # Exclude <bos> and <eos>