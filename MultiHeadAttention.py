import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 emb_dim:int=384,
                 head:int=3,
                 dropout:float=0.
    ):
        """
        引数:
            emb_dim : 埋め込み後のベクトル長
            head : ヘッドの数
            dropout : ドロップアウト率
        """

        super().__init__()
        self.head = head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // self.head
        self.sqrt_dh = self.head_dim**0.5 # D_hの二乗根, qk^Tを割るための係数
        
        # 入力をq,k,vに埋め込むための線形層
        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)

        # ドロップアウト層
        self.attn_drop = nn.Dropout(dropout)

        # MHSAの結果を出力に埋め込むための線形層
        self.w_o = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, z:torch.Tensor) -> torch.Tensor:
        """
        引数:
            z: MHSAのへの入力．shape => (B, N, D)
                B: batch_size, N: トークン数, D: 埋め込みベクトル長
        返り値:
            out: MHSAの出力. shape => (B, N, D)
        """
        batch_size, num_patch, _ = z.shape

        # 埋め込み
        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        # q, k, vをヘッドに分ける
        ## (B, N, D) -> (B, N, h, D/h)
        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)

        ## Self-Attentionができるように
        ## (B, N, h, D/h) -> (B, h, N, D/h)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # 内積
        ## (B, h, N, D/h) -> (B, h, D/h, N)
        k_T = k.transpose(2,3)
        ## (B, h, N, D/h) x (B, h, D/h, N) -> (B, h, N, N)
        dots = (q @ k_T) / self.sqrt_dh
        ## 列方向にソフトマックス
        attn = F.softmax(dots, dim=-1)
        ## ドロップアウト
        attn = self.attn_drop(attn)

        #　加重和
        ## (B, h, N, N) x (B, h, N, D/h) -> (B, h, N, D/h)
        out = attn @ v
        ## (B, h, N, D/h) -> (B, N, h, D/h)
        out = out.transpose(1,2)
        ## (B, N, h, D/h) -> (B, N, D)
        out = out.reshape(batch_size, num_patch, self.emb_dim)

        return out