import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch import optim
import numpy as np
import os
from functools import partial
import matplotlib.pyplot as plt, seaborn as sns

# ---- Optional: silence TensorFlow oneDNN informational logs (Lightning/TensorBoard may import TF) ----
# This avoids messages like:
# "oneDNN custom operations are on... To turn them off, set TF_ENABLE_ONEDNN_OPTS=0"
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# Reduce TensorFlow C++ log verbosity (0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
CHECKPOINT_PATH = "../saved_models/tutorial6"


def scaled_dot_product(q, k, v, mask=None, is_causal=False, dropout_p=0.0, training=True):
    # q shape (#, #, L, d)
    # k, v shape (#, #, S, d)
    # Map (S, d) to be (L, d)
    # Each l in [L]: Weighted sum of S vectors (d-dim) in v
    dk = q.size()[-1]  # (..., L, d)
    L, S = q.size(-2), k.size(-2)   # seq length of q and k

    # QK^T, shape [B, H, L, S] (for multi-head attention)
    attn_weights = (q @ k.transpose(-2, -1)) / (dk ** 0.5)

    # Use float math for stable softmax (esp. fp16/bf16)
    attn_weights = attn_weights.float()

    # Apply mask (supports broadcastable boolean masks like [B, 1, L, S] / [B, 1, S] / [L, S])
    # Convention: mask == True means "allowed to attend"
    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        attn_weights = attn_weights.masked_fill(~mask, float("-inf"))

    # Optional causal masking (disallow attending to future positions)
    if is_causal:
        causal_mask = torch.ones((L, S), dtype=torch.bool, device=q.device).triu(1)  # True where illegal
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

    attn_weights = F.softmax(attn_weights, dim=-1)  # last dim is a distribution
    attn_weights = attn_weights.to(dtype=q.dtype)

    attn_weights = F.dropout(attn_weights, p=dropout_p, training=training)

    output = attn_weights @ v
    ## Weighted sum of v's row vectors

    return output, attn_weights


def _demo_sdp_attention():
    demo_dtype = torch.float16 if device.type == "cuda" else torch.float32
    query = torch.rand(32, 8, 128, 64, dtype=demo_dtype, device=device)
    key = torch.rand(32, 8, 128, 64, dtype=demo_dtype, device=device)
    value = torch.rand(32, 8, 128, 64, dtype=demo_dtype, device=device)
    output, attn = scaled_dot_product(query, key, value, dropout_p=0.0, training=False)
    out = F.scaled_dot_product_attention(query, key, value)
    diff = (output - out).abs()
    print("SDPA diff max/mean:", diff.max().item(), diff.mean().item())


def expand_mask(mask):
    """
    Normalize attention masks to a shape broadcastable to attention weights [B, H, L, S].

    Supported inputs:
    - [L, S]          -> [1, 1, L, S]
    - [B, L, S]       -> [B, 1, L, S]
    - [B, 1, 1, S]    -> unchanged (key padding mask)
    - [B, 1, L, S]    -> unchanged
    - [B, H, L, S]    -> unchanged
    """
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional"
    if mask.ndim == 2:
        return mask.unsqueeze(0).unsqueeze(0)
    if mask.ndim == 3:
        return mask.unsqueeze(1)
    if mask.ndim == 4:
        return mask
    raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

class MultiheadAttention(nn.Module):
    def __init__(self, in_dim, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.qkv_proj = nn.Linear(in_features=in_dim, out_features=3 * emb_dim)
        self.o_proj = nn.Linear(in_features=emb_dim, out_features=in_dim)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1) # [Batch, Head, SeqLen, head_dim]

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask, dropout_p=0.0, training=self.training)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.emb_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o
    

class EncoderBlock(nn.Module):
    def __init__(self, in_dim, num_heads, dim_ff, dropout=0.0):
        super().__init__()
        self.multiheadAttn = MultiheadAttention(in_dim=in_dim, emb_dim=in_dim, num_heads=num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=dim_ff),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(in_features=dim_ff, out_features=in_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention block (with residual)
        x_attn = self.multiheadAttn(x, mask=mask)
        x = x + self.dropout(x_attn)
        x = self.norm1(x)

        # Feed-forward block (with residual)
        ff = self.mlp(x)
        x = x + self.dropout(ff)
        x = self.norm2(x)
        return x
    

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.multiheadAttn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x, mask=mask)
        return attention_maps


class PositionalEncoding(nn.Module):
    '''
    For dimension d, it creates d/2 angles:
    From frequency 1(fast) to 1/N(slow) with some large number N (e.g. 1e4)
    Sample d/2 values between 0..1: 0, 2/d, 4/d, ..., (d-2)/d
    So the frequency is N^(0), N^(-2/d), ..., N^(-1+2/d)
    '''
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Creates d_model/2 frequencies
        freqs = torch.exp((-np.log(1e4) / d_model) * torch.arange(0, d_model, 2).float())
        ### position * div_term is (max_len, 1) * (1, d_model/2) = (max_len, d_model/2)
        pe[:, 0::2] = torch.sin(position * freqs)
        pe[:, 1::2] = torch.cos(position * freqs)
        pe = pe.unsqueeze(0) #(1, max_len, d_model)

        # 確保 x 與 x.pe 要在同一個 device 上面
        # persistent=True: pe 會被寫入 model.state_dict()
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        # x: (batch_size, seq_len, emb_dim)
        x += self.pe[:, :x.size(1)]
        return x


    @torch.no_grad()
    def plot(self, max_len=None, figsize=(8, 3), cmap="RdGy", title="Positional encoding over hidden dimensions"):
        """
        視覺化 sin/cos positional encoding（教學/除錯用）。

        Args:
            max_len: 只畫前 max_len 個位置（None 表示用目前 buffer 的長度）
            figsize: matplotlib 圖大小
            cmap: imshow 色盤
            title: 圖表標題
        """
        pe = self.pe  # shape: (1, max_len, d_model)
        pe = pe[:, :max_len] if max_len is not None else pe

        # 轉成 (d_model, max_len) 方便畫 heatmap：y 軸是 hidden dim，x 軸是 position
        pe2d = pe.squeeze(0).T
        print("PE size:", pe2d.size())
        pe2d = pe2d.cpu().numpy()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        pos = ax.imshow(
            pe2d,
            cmap=cmap,
            extent=(1, pe2d.shape[1] + 1, pe2d.shape[0] + 1, 1)
        )
        fig.colorbar(pos, ax=ax)
        ax.set_xlabel("Position in sequence")
        ax.set_ylabel("Hidden dimension")
        ax.set_title(title)
        ax.set_xticks([1] + [i * 10 for i in range(1, 1 + pe2d.shape[1] // 10)])
        ax.set_yticks([1] + [i * 10 for i in range(1, 1 + pe2d.shape[0] // 10)])
        plt.show()


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]
    
    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    
    @classmethod
    @torch.no_grad()
    def plot_lr_factor(cls, warmup=100, max_iters=2000, base_lr=1e-3, figsize=(8, 3), title="Cosine Warm-up Learning Rate Scheduler"):
        """
        視覺化 cosine warmup 的 learning-rate factor 曲線（教學/除錯用）。

        說明：
        - 這裡為了建立 scheduler，需要一個 optimizer；因此用一個「假參數」來初始化 Adam。
        - 畫的是 factor（相對倍率），實際 lr = base_lr * factor。
        """
        # 用假參數建立 optimizer（只是為了初始化 _LRScheduler 所需的 optimizer/base_lrs）
        p = nn.Parameter(torch.empty(1))
        optimizer = optim.Adam([p], lr=base_lr)
        lr_scheduler = cls(optimizer=optimizer, warmup=warmup, max_iters=max_iters)

        epochs = list(range(max_iters))
        factors = [lr_scheduler.get_lr_factor(e) for e in epochs]

        sns.set()
        plt.figure(figsize=figsize)
        plt.plot(epochs, factors)
        plt.ylabel("Learning rate factor")
        plt.xlabel("Iterations (in batches)")
        plt.title(title)
        plt.show()
        sns.reset_orig()


class TransformerPredictor(pl.LightningModule):
    def __init__(self, input_dim, model_dim, num_classes, num_heads, num_layers, lr, warmup, max_iters, dropout=0.0, input_dropout=0.0, pad_idx=None):
        """
        Inputs:
            input_dim - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            num_layers - Number of encoder blocks to use.
            lr - Learning rate in the optimizer
            warmup - Number of warmup steps. Usually between 50 and 500
            max_iters - Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
            pad_idx - (optional) padding token id; used to ignore padding in loss/metrics for variable-length tasks
        """
        super().__init__()
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.input_dim, self.hparams.model_dim)
        )
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim)
        # Transformer
        self.transformer = TransformerEncoder(num_layers=self.hparams.num_layers,
                                              in_dim=self.hparams.model_dim,
                                              dim_ff=2*self.hparams.model_dim,
                                              num_heads=self.hparams.num_heads,
                                              dropout=self.hparams.dropout)
        # Output classifier per sequence lement
        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes)
        )

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)  # output: distribution over classes

        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError


class ReverseDataset(data.Dataset):
    def __init__(self, num_categories, seq_len, size):
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len
        self.size = size

        self.data = torch.randint(self.num_categories, size=(self.size, self.seq_len))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inp_data = self.data[idx]
        labels = torch.flip(inp_data, dims=(0,))
        return inp_data, labels
    

class VariableLengthReverseDataset(data.Dataset):
    """
    反轉序列任務（可變長度版本）：
    - 每筆 input 是長度 L 的整數 token 序列（L 隨機，2 <= L <= max_seq_len）
    - label 是 input 的反轉
    """
    def __init__(self, num_categories, min_seq_len, max_seq_len, size):
        super().__init__()
        assert min_seq_len >= 2, "min_seq_len should be >= 2"
        assert max_seq_len >= min_seq_len
        self.num_categories = num_categories
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 隨機抽一個長度
        L = torch.randint(low=self.min_seq_len, high=self.max_seq_len + 1, size=(1,)).item()
        inp = torch.randint(low=0, high=self.num_categories, size=(L,))  # 真實 token 只落在 [0, num_categories)
        labels = torch.flip(inp, dims=(0,))
        return inp, labels


def _pad_collate_reverse(batch, pad_idx):
    """
    將可變長度序列 padding 成同長度 batch，並產生 attention mask。

    Returns:
        inp_padded: [B, T]，padding 值為 0（並搭配 mask 才能分辨哪些是 PAD）
        lbl_padded: [B, T]，padding 值為 -100（CrossEntropyLoss 的 ignore_index）
        attn_mask : [B, 1, 1, T] bool，True 表示「允許 attend 到該 key 位置」（key padding mask）
    """
    inps, lbls = zip(*batch)
    lengths = torch.tensor([x.numel() for x in inps], dtype=torch.long)
    T = int(lengths.max().item())
    B = len(inps)

    # 我們用 0 當作 input padding value（注意：0 也是合法 token，所以一定要配合 mask 使用）
    inp_padded = torch.zeros((B, T), dtype=torch.long)
    # label padding 用 -100，配合 cross_entropy(ignore_index=-100) 忽略
    lbl_padded = torch.full((B, T), fill_value=-100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(inps, lbls)):
        L = x.numel()
        inp_padded[i, :L] = x
        lbl_padded[i, :L] = y

    # valid: [B, T]，True 表示該位置是有效 token
    valid = lbl_padded != -100
    # key padding mask: [B, 1, 1, T]
    # - 只遮蔽「key/value」端的 padding，避免任何 query attend 到 PAD
    # - 不遮蔽 query 端，避免在 padding query row 出現 "all -inf -> softmax NaN" 的問題
    attn_mask = valid[:, None, None, :]
    return inp_padded, lbl_padded, attn_mask


def _build_reverse_dataloaders(num_categories=10, min_seq_len=2, max_seq_len=16):
    """
    建立可變長度 ReverseTask 的 dataloaders（含 padding + attention mask）。
    """
    dataset = partial(VariableLengthReverseDataset, num_categories, min_seq_len, max_seq_len)
    # padding 行為在 collate_fn 裡固定（input padding=0, label padding=-100）
    collate_fn = partial(_pad_collate_reverse, pad_idx=None)

    train_loader = data.DataLoader(
        dataset(50000),
        batch_size=128,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = data.DataLoader(dataset(1000), batch_size=128, collate_fn=collate_fn)
    test_loader = data.DataLoader(dataset(10000), batch_size=128, collate_fn=collate_fn)

    # 印出一筆「未 padding 前」的原始樣本（可變長度）
    inp_data, labels = train_loader.dataset[0]
    print("Input data:", inp_data)
    print("Labels:    ", labels)
    return train_loader, val_loader, test_loader


class ReversePredictor(TransformerPredictor):
    def _calculate_loss(self, batch, mode="train"):
        # Fetch data and transform categories to one-hot vectors
        # batch 可能是：
        # - 固定長度版本: (inp_tokens, labels)
        # - 可變長度版本: (inp_tokens, labels, attn_mask)
        if len(batch) == 2:
            inp_tokens, labels = batch
            attn_mask = None
        else:
            inp_tokens, labels, attn_mask = batch

        # inp_tokens: [B, T] 的整數 token id
        # 轉成 one-hot features: [B, T, C]
        inp_data = F.one_hot(inp_tokens, num_classes=self.hparams.num_classes).float()

        # 若有 attention mask（key padding mask），把 padding 位置的 one-hot 歸零，避免把 padding 當作有效 token
        if attn_mask is not None:
            # attn_mask: [B, 1, 1, T] -> valid: [B, T]
            valid = attn_mask.squeeze(1).squeeze(1)
            inp_data = inp_data * valid[:, :, None].float()

        # Perform prediction and calculate loss and accuracy
        preds = self.forward(inp_data, mask=attn_mask, add_positional_encoding=True)  # [B, T, C]

        # padding 位置不應該算進 loss/acc：我們用 labels == -100 當作 padding（ignore_index）
        loss = F.cross_entropy(
            preds.view(-1, preds.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        valid = labels != -100  # [B, T]
        correct = (preds.argmax(dim=-1) == labels) & valid
        acc = correct.float().sum() / valid.float().sum().clamp_min(1.0)

        # Logging
        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")


@torch.no_grad()
def _preview_reverse_samples(dataloader, n_samples=3, title="Sample preview"):
    """
    以「人類可讀」的方式印出幾筆 ReverseDataset 的 (input, label)。
    - input: 一串類別 id（整數）
    - label: input 的反轉序列（目標答案）
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    ds = dataloader.dataset
    n = min(n_samples, len(ds))
    for i in range(n):
        inp, lbl = ds[i]
        print(f"[{i}] input : {inp.tolist()}")
        print(f"    label : {lbl.tolist()}  (should equal input reversed)")


@torch.no_grad()
def _trace_reverse_workflow(model: TransformerPredictor, batch, add_positional_encoding=True, show_examples=2):
    """
    追蹤一個 batch 在模型內部被轉換的過程（初學者導覽）。

    我們會印出：
    1) 原始 input/label（整數類別）
    2) one-hot 後的 features 長什麼樣子
    3) 經過 input_net / positional_encoding / transformer encoder / output_net 後的張量形狀
    4) 最終預測（argmax）與正確答案的對照
    5) attention maps 的形狀（每層一個 [B, Heads, T, T]）
    """
    model.eval()

    # batch 可能是：
    # - 固定長度版本: (inp_tokens, labels)
    # - 可變長度版本: (inp_tokens, labels, attn_mask)
    if len(batch) == 2:
        inp_tokens, labels = batch
        attn_mask = None
    else:
        inp_tokens, labels, attn_mask = batch

    # inp_tokens: [B, T] int64, labels: [B, T]
    # attn_mask (optional): bool mask broadcastable to [B, H, L, S]
    # - variable-length version uses key padding mask: [B, 1, 1, T]
    inp_tokens = inp_tokens.to(model.device)
    labels = labels.to(model.device)
    attn_mask = attn_mask.to(model.device) if attn_mask is not None else None

    B, T = inp_tokens.shape
    C = model.hparams.num_classes  # 類別數（可變長度版本會包含 PAD）
    pad_idx = getattr(model.hparams, "pad_idx", None)

    print("\n" + "=" * 80)
    print("Workflow trace (one batch)")
    print("=" * 80)
    print(f"Raw tokens shape: {tuple(inp_tokens.shape)}  (B, T) = ({B}, {T})")
    print(f"Labels     shape: {tuple(labels.shape)}")
    # 可變長度版本：labels == -100 表示 padding
    valid_from_labels = (labels != -100)
    lengths = valid_from_labels.long().sum(dim=1)
    print(f"lengths(min/mean/max): {int(lengths.min())}/{lengths.float().mean().item():.2f}/{int(lengths.max())}")

    # (1) 整數類別 -> one-hot features
    # one_hot: [B, T, C]，每個 token 會變成一個長度為 C 的向量（只有對應類別的位置為 1）
    one_hot = F.one_hot(inp_tokens, num_classes=C).float()
    if attn_mask is not None:
        valid = attn_mask.squeeze(1).squeeze(1)
        one_hot = one_hot * valid[:, :, None].float()
    print(f"One-hot features shape: {tuple(one_hot.shape)}  (B, T, C) where C={C}")

    # 示範看一兩筆資料（避免一次印太多）
    n_show = min(show_examples, B)
    for i in range(n_show):
        print(f"\nExample {i}:")
        print(f"  input tokens : {inp_tokens[i].tolist()}")
        print(f"  target label : {labels[i].tolist()}")
        # 看看第一個 token 的 one-hot 向量長什麼樣子（應該只有一個位置是 1）
        first_token_vec = one_hot[i, 0].tolist()
        print(f"  one-hot(x[0]) (len={C}): {first_token_vec}")

    # (2) input_net：把 C 維 one-hot 投影到 model_dim 維度（embedding）
    x_in = model.input_net(one_hot.to(model.device))  # [B, T, model_dim]
    print(f"\nAfter input_net shape: {tuple(x_in.shape)}  (B, T, model_dim)")

    # (3) positional encoding：加入位置資訊（仍是 [B, T, model_dim]）
    if add_positional_encoding:
        x_pe = model.positional_encoding(x_in.clone())
        print(f"After positional_encoding shape: {tuple(x_pe.shape)}  (B, T, model_dim)")
    else:
        x_pe = x_in
        print("Positional encoding skipped (add_positional_encoding=False)")

    # (4) transformer encoder：多層 self-attention + FFN，輸出仍是序列表示 [B, T, model_dim]
    x_enc = model.transformer(x_pe, mask=attn_mask)  # [B, T, model_dim]
    print(f"After transformer encoder shape: {tuple(x_enc.shape)}  (B, T, model_dim)")

    # (5) output_net：對每個位置做分類，得到 logits: [B, T, C]
    logits = model.output_net(x_enc)
    print(f"After output_net (logits) shape: {tuple(logits.shape)}  (B, T, num_classes)")

    # (6) 轉成預測類別（每個位置取 argmax）
    preds = logits.argmax(dim=-1)  # [B, T]
    print(f"Pred tokens shape: {tuple(preds.shape)}  (B, T)")

    for i in range(n_show):
        print(f"\nPrediction example {i}:")
        print(f"  pred : {preds[i].tolist()}")
        print(f"  gold : {labels[i].tolist()}")
        exact = bool((preds[i] == labels[i]).all().item())
        print(f"  exact match: {exact}")

    # (7) 注意力矩陣（每層一個）
    # attention_maps 是 list，長度 = num_layers；每個 attention: [B, Heads, T, T]
    attention_maps = model.get_attention_maps(one_hot.to(model.device), mask=attn_mask, add_positional_encoding=add_positional_encoding)
    print(f"\nNumber of layers (attention maps): {len(attention_maps)}")
    if len(attention_maps) > 0:
        print(f"Attention map[0] shape: {tuple(attention_maps[0].shape)}  (B, Heads, T, T)")


@torch.no_grad()
def _manual_reverse_test(model: TransformerPredictor, seq_len=16, num_classes=10, add_positional_encoding=True):
    """
    手動造一筆測試 input，看看模型預測是否為反轉序列。
    """
    model.eval()

    # 手動造一筆「可變長度」輸入：先選一個 L，再 padding 到 seq_len
    L = torch.randint(low=2, high=seq_len + 1, size=(1,)).item()
    tokens = torch.randint(low=0, high=num_classes, size=(L,), device=model.device)

    # input padding 用 0（但會配合 attn_mask 把 padding one-hot 歸零）
    inp = torch.zeros((1, seq_len), device=model.device, dtype=torch.long)
    inp[0, :L] = tokens

    # gold padding 用 -100（ignore_index）
    gold = torch.full((1, seq_len), fill_value=-100, device=model.device, dtype=torch.long)
    gold[0, :L] = torch.flip(tokens, dims=(0,))

    # attention mask：避免 attend 到 PAD（key padding mask）
    valid = torch.zeros((1, seq_len), device=model.device, dtype=torch.bool)
    valid[0, :L] = True
    attn_mask = valid[:, None, None, :]  # [1, 1, 1, T]

    # 跟訓練一樣，把 token 轉成 one-hot features，並把 padding 位置歸零
    one_hot = F.one_hot(inp, num_classes=num_classes).float() * valid[:, :, None].float()
    logits = model(one_hot, mask=attn_mask, add_positional_encoding=add_positional_encoding)  # [1, T, C]
    pred = logits.argmax(dim=-1)  # [1, T]

    print("\n" + "=" * 80)
    print("Manual test (single sequence)")
    print("=" * 80)
    print(f"length (L): {L} | label pad: -100")
    print(f"input: {inp.squeeze(0).tolist()}")
    print(f"gold : {gold.squeeze(0).tolist()}  (input reversed; PAD ignored)")
    print(f"pred : {pred.squeeze(0).tolist()}")
    valid_pos = gold != -100
    exact = bool(((pred == gold) | ~valid_pos).all().item())
    print(f"exact match (ignoring PAD): {exact}")


def train_reverse(**kwargs):
    # 這個函式做的事：
    # 1) 建立 Lightning Trainer（訓練控制器）與 checkpoint 目錄
    # 2) 若已經有訓練好的 checkpoint，直接載入；否則就訓練一個新模型
    # 3) 在 val/test dataloader 上評估，回傳模型與成績

    # 設定：把這個任務（ReverseTask）的所有輸出放在 CHECKPOINT_PATH/ReverseTaskVarLenV4 下面
    # 之所以用不同資料夾，是因為我們把 padding 策略改成：
    # - input padding 用 0 + one-hot 歸零
    # - label padding 用 -100 (ignore_index)
    root_dir = os.path.join(CHECKPOINT_PATH, "ReverseTaskVarLenV4")  # 例如 ../saved_models/tutorial6/ReverseTaskVarLenV4

    # 若資料夾不存在就建立；exist_ok=True 表示「已存在也不會報錯」
    os.makedirs(root_dir, exist_ok=True)

    # 準備 checkpoint callback：保存「驗證集表現最佳」的模型
    # - dirpath：存檔資料夾
    # - filename：存檔名稱（會產生 root_dir/best.ckpt）
    # - monitor="val_acc"：以 validation_step 記錄的 val_acc 當作指標
    checkpoint_cb = ModelCheckpoint(
        dirpath=root_dir,
        filename="best",
        save_weights_only=True,
        mode="max",
        monitor="val_acc",
    )

    # 建立 PyTorch Lightning 的 Trainer：負責跑訓練/驗證/測試迴圈（你不用自己寫 for-epoch/for-batch）
    trainer = pl.Trainer(
        default_root_dir=root_dir,  # Lightning 的 log、checkpoint 等預設輸出位置
        callbacks=[
            # 每個 epoch 結束後，根據 monitor 指標（這裡是 val_acc）決定是否要存「目前最佳」的權重
            # save_weights_only=True 表示只存模型權重（較小）；mode="max" 表示 val_acc 越大越好
            checkpoint_cb
        ],
        # 選擇用 GPU 或 CPU；device 是你在檔案上面用 torch.cuda.is_available() 判斷出來的
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,          # 使用 1 張 GPU（或 1 個 CPU 裝置）
        max_epochs=20,      # 可變長度版本通常更難一點，訓練久一點比較容易收斂
        gradient_clip_val=5 # 梯度裁剪：避免梯度爆炸（尤其是 Transformer 類模型）
    )

    # Lightning 會預設記錄一個 hp_metric（超參數的指標）；這裡把它關掉，避免額外欄位干擾 log
    trainer.logger._default_hp_metric = None

    # 準備：上次訓練存下來的「最佳模型」checkpoint 檔案路徑
    # 注意：這裡要跟上面的 ModelCheckpoint(dirpath, filename) 對齊，否則永遠找不到檔案
    pretrained_filename = os.path.join(root_dir, "best.ckpt")

    # 如果 checkpoint 檔案存在，就直接載入模型並跳過訓練（省時間）
    # 若載入失敗（常見原因：超參數/維度已改動），就退回重新訓練
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        try:
            model = ReversePredictor.load_from_checkpoint(pretrained_filename)
        except Exception as e:
            print(f"Checkpoint load failed (will retrain). Reason: {type(e).__name__}: {e}")
            model = None
    else:
        model = None

    if model is None:
        # 若沒有可用 checkpoint：建立一個新模型準備訓練
        #
        # max_iters 是給 CosineWarmupScheduler 用的「總訓練步數（總 batch 數）」
        # len(train_loader) = 每個 epoch 的 batch 數
        # trainer.max_epochs * len(train_loader) = 總步數
        model = ReversePredictor(max_iters=trainer.max_epochs * len(train_loader), **kwargs)

        # 開始訓練：Trainer 會自動呼叫 model.training_step / model.validation_step
        trainer.fit(model, train_loader, val_loader)

    # 若剛剛有跑訓練，checkpoint_cb.best_model_path 會記錄「val_acc 最佳」那個 ckpt 的路徑
    # 我們用它來確保後面評估的是最佳模型（而不是最後一個 epoch 的模型）
    best_ckpt_path = checkpoint_cb.best_model_path
    if best_ckpt_path:
        model = ReversePredictor.load_from_checkpoint(best_ckpt_path)

    # 評估：
    # - validation：用 trainer.validate(...) 跑的是 model.validation_step，回傳的 key 會是 "val_acc"/"val_loss"
    # - test：用 trainer.test(...) 跑的是 model.test_step，回傳的 key 會是 "test_acc"/"test_loss"
    val_result = trainer.validate(model, val_loader, verbose=False)  # list[dict]，通常長度為 1
    test_result = trainer.test(model, test_loader, verbose=False)    # list[dict]，通常長度為 1

    # 把結果整理成你要回傳的 dict（方便後續 print / logging）
    result = {
        "val_acc": val_result[0]["val_acc"],
        "test_acc": test_result[0]["test_acc"],
    }

    # 把模型搬到你指定的 device（GPU/CPU）；方便你後續手動做推論/視覺化等
    model = model.to(device)

    # 回傳：訓練好的模型 + 成績 dict
    return model, result


if __name__ == "__main__":
    # _demo_sdp_attention()
    # PositionalEncoding(d_model=48, max_len=96).plot()
    # CosineWarmupScheduler.plot_lr_factor(warmup=100, max_iters=2000, base_lr=1e-3)

    train_loader, val_loader, test_loader = _build_reverse_dataloaders(num_categories=10, min_seq_len=2, max_seq_len=16)

    # (教學/導覽) 先看幾筆資料長什麼樣子
    _preview_reverse_samples(train_loader, n_samples=3, title="Training samples preview (VariableLengthReverseDataset)")

    reverse_model, reverse_result = train_reverse(
        # 可變長度版本：input_dim/num_classes 都等於 num_categories（padding 位置用 mask 與 label_pad=-100 處理）
        input_dim=train_loader.dataset.num_categories,
        # 可變長度反轉比固定長度更需要模型容量（要先推斷長度，再對齊位置）
        model_dim=64,
        num_heads=4,
        num_classes=train_loader.dataset.num_categories,
        num_layers=2,
        dropout=0.0,
        lr=5e-4,
        warmup=50,
    )

    print(f"Val accuracy:  {(100.0 * reverse_result['val_acc']):4.2f}%")
    print(f"Test accuracy: {(100.0 * reverse_result['test_acc']):4.2f}%")

    # (教學/導覽) 用一個 batch 追蹤資料在模型裡如何被轉換（包含 attention map 的形狀）
    batch = next(iter(val_loader))
    _trace_reverse_workflow(reverse_model, batch, add_positional_encoding=True, show_examples=2)

    # (教學/導覽) 手動造一筆測試 input，看模型是否能預測出反轉序列
    _manual_reverse_test(
        reverse_model,
        seq_len=train_loader.dataset.max_seq_len,
        num_classes=train_loader.dataset.num_categories,
        add_positional_encoding=True,
    )