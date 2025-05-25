import os
import sys
import inspect
sys.path.append(os.getcwd())

import torch
from torch import nn
import torch.nn.functional as F

from src.handmade_trf import (
    TrfConfig, 
    FlashSelfAttention,
    FeedForward,
)


class GPTBlock(nn.Module):

    def __init__(self, cfg: TrfConfig):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(cfg.emb_dim)
        self.attn = FlashSelfAttention(cfg)  # Flash-attn - оптимизация soft-max, заточенная под работу CUDA (~ +3% tok/s)
        self.layer_norm2 = nn.LayerNorm(cfg.emb_dim)
        self.ffn = FeedForward(cfg)

    
    def forward(self, idx):
        x = idx + self.attn(self.layer_norm1(idx))
        x = x + self.ffn(self.layer_norm2(idx))
        return x
    

class GPT2(nn.Module):

    def __init__(self, cfg: TrfConfig):
        super().__init__()

        self.cfg = cfg

        self.trf = nn.ModuleDict(dict(
            token_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim),
            pos_emb = nn.Embedding(cfg.context_size, cfg.emb_dim),
            heads = nn.ModuleList([GPTBlock(cfg) for _ in range(cfg.n_blocks)]),
            layer_norm = nn.LayerNorm(cfg.emb_dim),
        ))
        self.classifier = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

        self.trf.token_emb.weight = self.classifier.weight  # Оптимизация количества параметров и затухания градиента

        self.apply(self._init_weights)

    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0., std=0.02)
            
            if hasattr(module, "bias") and (module.bias is not None):
                nn.init.zeros_(module.bias)


    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.size()
        
        assert T <= self.cfg.context_size

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.trf.token_emb(idx) + self.trf.pos_emb(pos)
        for block in self.trf.heads:
            x = block(x)
        x = self.trf.layer_norm(x)
        
        logits = self.classifier(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {name: p for name, p in self.named_parameters() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"

        return torch.optim.AdamW(
            optim_groups, 
            lr=learning_rate, 
            betas=(0.9, 0.95), 
            eps=1e-8, 
            fused=use_fused,
        )
    
    
    def generate(self, tokens: list, num_return_sequences: int = 4, max_length: int = 32, top_k: int = 50):
        self.eval()
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = self(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        
        return xgen[:, :max_length]


class RandomSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, data, context_size):
        self.data = data
        self.context_size = context_size

    
    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        # просто игнорируем idx и семплируем случайно
        i = torch.randint(0, len(self.data) - self.context_size - 1, (1,)).item()
        x = self.data[i : i+self.context_size]
        y = self.data[i+1 : i+1+self.context_size]
        
        return x, y
    

if __name__ == "__main__":
    import math
    from time import time

    from tqdm import tqdm

    from src.handmade_tokenizer import Tokenizer
    
    
    torch.manual_seed(42)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        torch.cuda.manual_seed(42)
    device_type = device.split(":")[0]

    print(f"{device=} | {device_type=}")

    torch.set_float32_matmul_precision("high")  # Проведение операций не с F32 типом данных, а с TF32 (~ +200-300% tok/s)
    
    root = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    
    # Инициализация токенизатора
    tokenizer = Tokenizer(os.path.join(root, "data", "tokenizer.json"))

    # Работа с гиперпараметрами
    vocab_size = len(tokenizer.vocab)
    while vocab_size % 2**4 != 0:  # В идеале vocab_size должен делиться на как можно большее 2**n для эффективного использования cuda
        vocab_size += 1  # В модель будут добавлены фиктивные нейроны, которые быстро занулятся, т.к. в данных нет таких токенов
    
    context_size = 64
    batch_size = 16
    emb_dim = 256
    n_blocks = 12
    n_heads = 8

    cfg = TrfConfig(
        emb_dim=emb_dim,
        context_size=context_size,
        head_size=emb_dim//n_heads,
        n_heads=n_heads,
        n_blocks=n_blocks,
        vocab_size=vocab_size,
    )

    # Работа с данными
    with open(os.path.join(root, "data", "tinyshakespeare.txt"), "r", encoding="utf-8") as f:
        text = f.read()
    
    data = tokenizer.encode(text)
    n = int(0.9 * len(data))

    train_dataset = RandomSequenceDataset(torch.tensor(data[:n], dtype=torch.long), context_size=context_size)
    val_dataset = RandomSequenceDataset(torch.tensor(data[n:], dtype=torch.long), context_size=context_size)

    dataloader_params = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, **dataloader_params)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, **dataloader_params)

    print(f"Всего токенов: {len(data)} | объем train: {n}")
    
    token_batch_size = 2 ** 12  # ~4k токенов в батче
    grad_accum_steps = token_batch_size // (batch_size * context_size)

    # Инициализация модели
    model = GPT2(cfg).to(device)
    model = torch.compile(model)  # Оптимизация графа и уменьшение накладных расходов интерпретатора (~ +30-40% tok/s)

    print(f"Всего параметров: {sum(p.numel() for p in model.parameters())}")

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    epoches = 3
    max_steps = epoches * len(data) // token_batch_size  # Шагов за эпоху
    warmup_steps = max_steps // 12
    
    def get_lr(it):
        
        if it < warmup_steps:  # Линейный рост lr на прогреве
            return max_lr * (it + 1) / warmup_steps
        
        if it > max_steps:  # После lr_decay_iters затухающих шагов
            return min_lr
        
        # После прогрева косинусное затухание, сходящееся к min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    # Инициализация оптимизатора
    optimizer = model.configure_optimizers(
        weight_decay=0.1, 
        learning_rate=6e-4,
        device_type=device_type,
    )

    # train-loop
    for step in range(max_steps):

        start = time()

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.
        for micro_step in range(grad_accum_steps):  # аккумулированный градиент для достижения token_batch_size токенов за один шаг оптимизации
                                                    # используется в условиях ограниченности cuda ресурсов
            x, y = next(iter(train_dataloader))
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):  # некоторые допустимые операции будут производиться в torch.bfloat16 для ускорения
                logits, loss = model(x, y)
            
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()

            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()  # ожидание завершения работы cuda
        
        end = time()
        dt = end - start
        tokens_per_sec = batch_size * context_size * grad_accum_steps / (end - start)
        
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

        # Валидация
        if step % 100 == 0:

            model.eval()
            with torch.no_grad():
                losses = []
                for x, y in tqdm(val_dataloader):
                    x, y = x.to(device), y.to(device)

                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    
                    losses.append(loss.item())

                val_loss = sum(losses) / len(losses)
                print(f"step {step:5d} | val_loss: {val_loss:.6f}")

                if step % 200 and step > 0:
                    checkpoint_path = os.path.join(root, "data", f"model_{step:05d}.pt")
                    checkpoint = {
                        "model": model.state_dict(),
                        "config": model.cfg,
                        "step": step, 
                        "val_loss": val_loss,
                    }

                    torch.save(checkpoint, checkpoint_path)

            if step % 200 and step > 0:
                # Проверка генерации
                tokens = tokenizer.encode("First Citizen:")
                xgen = model.generate(tokens)
                for i in range(len(xgen)):
                    decoded = tokenizer.decode(xgen[i].tolist())
                    print(f"sample {i}: {decoded}")
