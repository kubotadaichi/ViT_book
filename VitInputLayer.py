import torch
import torch.nn as nn

class VitInputLayer(nn.Module):
    def __init__(self, 
                 in_channels:int=3,
                 emb_dim:int=384,
                 num_patch_row:int=2,
                 image_size:int=32
                ):
        """
        in_channels : 入力画像のチャンネル数
        emb_dim : 埋め込み後のベクトルの長さ
        num_patch : 高さ方向のパッチの数
        image_size : 入力画像の1辺の長さ，入力画像の高さと幅は同じであると仮定
        """

        super().__init__()
        self.in_channels=in_channels
        self.emb_dim = emb_dim
        self.num_patch_row = num_patch_row
        self.image_size = image_size

        self.num_patch = self.num_patch_row ** 2

        # パッチの大きさ
        # 例 : 入力画像の１辺の長さが32， patch_size_row=2の場合, patch_size = 16
        self.patch_size = int(self.image_size // self.num_patch_row)

        # 入寮画像のパッチへの分割 & パッチの埋め込みを一気に行う層 => kernelを学習することを埋め込みと呼んでいる？
        self.patch_emb_layer = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # クラストークン
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, emb_dim)
        )

        # 位置埋め込み
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.num_patch+1, emb_dim)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        引数:
            x : 入力画像.shape => (B, C, H, W)
                B: batch_size, C: num_channels, H: height, W: width
        返り値:
            z_0: ViTへの入力.shape => (B, N, D)
                B: batch_size, N: num_token, D: emb_dim
        """

        # バッチの埋め込み & flatten
        ## (B, C, H, W) -> (B, D, H/P, W/P)
        z_0 = self.patch_emb_layer(x)
        ## (B, D, H/P, W/P) -> (B, D, Np)
        z_0 = z_0.flatten(2)
        ## 軸の入れ替え (B, D, Np) -> (B, Np, D)
        z_0 = z_0.transpose(1,2)

        # バッチの埋め込みの先頭に暮らすトークンを結合
        ## (B, Np, D) -> (B, N, D)
        z_0 = torch.cat(
            [self.cls_token.repeat(repeats=(x.size(0),1,1)), z_0], dim=1
        )

        # 位置埋め込みの加算
        ## (B, N, D) -> (B, N, D)
        z_0 = z_0 + self.pos_emb

        return z_0

def main():
    batch_size, channel, height, width = 2, 3, 32, 32
    x = torch.randn(batch_size, channel, height, width)
    input_layer = VitInputLayer()
    z_0 = input_layer(x)
    print(z_0.shape)

if __name__ == "__main__":
    main()