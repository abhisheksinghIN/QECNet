"""
soa.py - model factory for experiments

- Provides a generic PyTorch UNet implemented below (no segmentation-models-pytorch needed for UNet).
- Keeps DeepLabV3+ using segmentation_models_pytorch (if installed).
- Provides robust wrappers for SegFormer and TransUNet with helpful error messages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Attempt to import segmentation_models_pytorch for DeepLabV3+ (optional)
try:
    import segmentation_models_pytorch as smp
    _HAS_SMP = True
except Exception:
    _HAS_SMP = False


from transformers import SegformerForSemanticSegmentation, SegformerConfig

class SegFormer(nn.Module):
    def __init__(self, num_classes=9, in_channels=3):
        super(SegFormer, self).__init__()
        
        # Create a Segformer configuration with specific number of classes
        config = SegformerConfig(num_labels=num_classes)
        
        # Initialize the Segformer model for semantic segmentation
        self.segformer = SegformerForSemanticSegmentation(config)
        
        # Adjust the input layer to accept the specified number of input channels
        old_conv = self.segformer.segformer.encoder.patch_embeddings[0].proj
        self.segformer.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        )
    
    def forward(self, x):
        # Forward pass through the Segformer model for semantic segmentation
        outputs = self.segformer(pixel_values=x)
        
        # Extract the logits for semantic segmentation
        logits = outputs.logits
        
        # Upsample the output to match the target size
        x = F.interpolate(logits, size=(256, 256), mode='bilinear', align_corners=False)
        return x


## -------------------------
## UNet building blocks
## -------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=4):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024 + 512, 512)
        self.up2 = Up(512 + 256, 256)
        self.up3 = Up(256 + 128, 128)
        self.up4 = Up(128 + 64, 64)
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        #print(f"Input: {x.shape}")
        x1 = self.inc(x)
        #print(f"x1: {x1.shape}")
        x2 = self.down1(x1)
        #print(f"x2: {x2.shape}")
        x3 = self.down2(x2)
        #print(f"x3: {x3.shape}")
        x4 = self.down3(x3)
        #print(f"x4: {x4.shape}")
        x5 = self.down4(x4)
        #print(f"x5: {x5.shape}")
        
        x = self.up1(x5, x4)
        #print(f"up1: {x.shape}")
        x = self.up2(x, x3)
        #print(f"up2: {x.shape}")
        x = self.up3(x, x2)
        #print(f"up3: {x.shape}")
        x = self.up4(x, x1)
        #print(f"up4: {x.shape}")
        logits = self.outc(x)
        #print(f"Output: {logits.shape}")
        return logits


# ---------------------------------------------------------
#  Factory
# ---------------------------------------------------------
def build_unet(in_channels=4, num_classes=6):
    return UNet(in_channels=in_channels, num_classes=num_classes)


def build_deeplabv3(num_classes=4, in_channels=3):
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,        # ? TRAIN FROM SCRATCH
        in_channels=in_channels,
        classes=num_classes
    )
    return model

# networks/transunet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Small Transformer blocks (vanilla)
# ----------------------
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, D)
        x_res = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = x_res + self.dropout(attn_out)

        x_res = x
        x = self.norm2(x)
        x = x_res + self.mlp(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


# ----------------------
# CNN encoder / decoder building blocks
# ----------------------
def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride, bias=False)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            conv3x3(out_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """Conv block with optional downsample (maxpool)."""
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.conv = DoubleConv(in_ch, out_ch)
        if self.downsample:
            self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        if self.downsample:
            return self.pool(x), x  # return pooled, skip
        else:
            return x, x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv = DoubleConv(in_ch + skip_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv((in_ch // 2) + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if necessary
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


# ----------------------
# TransUNet - compact, from scratch (no pretrained)
# ----------------------
class TransUNet(nn.Module):
    def __init__(
        self,
        in_channels=4,
        num_classes=6,
        img_size=256,
        vit_embed_dim=512,
        vit_depth=4,
        vit_heads=8,
        vit_mlp_ratio=4.0,
        vit_dropout=0.0,
        bilinear=True
    ):
        """
        img_size: expected input spatial size (e.g. 256). Must be divisible by 16.
        vit_embed_dim: embedding dim for transformer (project conv features to this dim)
        vit_depth: number of transformer layers
        vit_heads: number of attention heads
        """
        super().__init__()

        if img_size % 16 != 0:
            raise ValueError("img_size must be divisible by 16 (we downsample by 16).")

        # ----- Encoder (CNN) -----
        self.inc = DoubleConv(in_channels, 64)
        self.enc1 = EncoderBlock(64, 128, downsample=True)   # -> H/2
        self.enc2 = EncoderBlock(128, 256, downsample=True)  # -> H/4
        self.enc3 = EncoderBlock(256, 512, downsample=True)  # -> H/8
        self.enc4 = EncoderBlock(512, 512, downsample=True)  # -> H/16

        # channels at the ViT input
        self.vit_in_ch = 512
        self.vit_h = img_size // 16
        self.vit_w = img_size // 16
        self.num_tokens = self.vit_h * self.vit_w

        # Project conv features -> transformer embedding dim
        self.project_to_embedding = nn.Conv2d(self.vit_in_ch, vit_embed_dim, kernel_size=1)

        # positional embedding and transformer
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, vit_embed_dim))
        self.transformer = TransformerEncoder(
            dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            mlp_ratio=vit_mlp_ratio,
            dropout=vit_dropout
        )

        # project back from embedding -> decoder channel count
        self.project_back = nn.Conv2d(vit_embed_dim, self.vit_in_ch, kernel_size=1)

        # ----- Decoder (UNet style) -----
        self.dec1 = DecoderBlock(self.vit_in_ch, 512, 256, bilinear=bilinear)
        self.dec2 = DecoderBlock(256, 256, 128, bilinear=bilinear)
        self.dec3 = DecoderBlock(128, 128, 64, bilinear=bilinear)
        self.dec4 = DecoderBlock(64, 64, 64, bilinear=bilinear)

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, H, W = x.shape

        # ---- ENCODER ----
        x0 = self.inc(x)
        x1_pool, x1 = self.enc1(x0)
        x2_pool, x2 = self.enc2(x1_pool)
        x3_pool, x3 = self.enc3(x2_pool)

        # 4th downsample only if input is large enough
        if x3_pool.shape[2] > 16:
            x4_pool, x4 = self.enc4(x3_pool)
        else:
            x4 = x3_pool

        h, w = x4.shape[2], x4.shape[3]
        self.vit_h, self.vit_w = h, w
        self.num_tokens = h * w

        # ---- VIT ----
        vit_feat = self.project_to_embedding(x4)
        B, D, h, w = vit_feat.shape
        tokens = vit_feat.flatten(2).transpose(1, 2)  # (B,N,D)

        # resize positional embedding if needed
        if self.pos_embed.shape[1] != self.num_tokens:
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.num_tokens, self.pos_embed.shape[-1]).to(x.device)
            )

        tokens = tokens + self.pos_embed
        tokens = self.transformer(tokens)
        tokens = tokens.transpose(1, 2).reshape(B, D, h, w)
        vit_out = self.project_back(tokens)

        # ---- DECODER ----
        if x4 is x3_pool:
            # 3-level UNet (input small)
            d1 = self.dec2(vit_out, x2)
            d2 = self.dec3(d1, x1)
            d3 = self.dec4(d2, x0)
            logits = self.head(d3)
        else:
            # 4-level UNet (input large)
            d1 = self.dec1(vit_out, x3)
            d2 = self.dec2(d1, x2)
            d3 = self.dec3(d2, x1)
            d4 = self.dec4(d3, x0)
            logits = self.head(d4)

        return logits
        
        

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


# -----------------------------
# Global config
# -----------------------------
GLOBAL_N_WIRES = 6  # Quantum circuit wires


# -----------------------------
# Quantum Convolutional Layer
# -----------------------------
class QuantumConv(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size=2, n_layers=4):
        super().__init__()
        self.n_wires = GLOBAL_N_WIRES
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        dev = qml.device("default.qubit", wires=self.n_wires)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_wires))
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        weight_shapes = {"weights": (n_layers, self.n_wires, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        self.fc = nn.Linear(self.n_wires, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        # unfold patches
        x_unfold = x.unfold(2, p, p).unfold(3, p, p)
        x_unfold = x_unfold.contiguous().view(B, C, -1, p * p)
        x_mean = x_unfold.mean(dim=-1).permute(0, 2, 1).reshape(-1, C)

        # pad or truncate to n_wires
        if C < self.n_wires:
            pad = torch.zeros((x_mean.size(0), self.n_wires - C),
                              device=x.device, dtype=x_mean.dtype)
            x_in = torch.cat([x_mean, pad], dim=1)
        else:
            x_in = x_mean[:, :self.n_wires]

        # quantum layer
        q_out = torch.vmap(self.q_layer)(x_in)
        q_out = self.fc(q_out)
        q_out = self.norm(q_out)
        q_out = torch.relu(q_out)

        # reshape back to image
        q_out = q_out.view(B, H // p, W // p, self.out_channels).permute(0, 3, 1, 2)
        q_out = F.interpolate(q_out, scale_factor=p, mode='bilinear', align_corners=False)
        return q_out


# -----------------------------
# ConvBlock
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True):
        super().__init__()
        self.down = down
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if down:
            self.pool = nn.MaxPool2d(2)
        else:
            self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.down:
            return x, self.pool(x)
        else:
            return self.up(x)


# -----------------------------
# Quantum U-Net Block
# -----------------------------
class QBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True):
        super().__init__()
        self.down = down
        self.q1 = QuantumConv(in_channels, out_channels)
        self.q2 = QuantumConv(out_channels, out_channels)
        if not down:
            self.resample = nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2)
        else:
            self.resample = None

    def forward(self, x):
        x = self.q1(x)
        x = self.q2(x)
        if self.down:
            return x, F.avg_pool2d(x, 2, 2)
        else:
            return self.resample(x)


# -----------------------------
# Quantum U-Net
# -----------------------------
class QuantumUNet(nn.Module):
    def __init__(self, in_channels=15, num_classes=6):
        super().__init__()

        # ----- Encoder -----
        self.enc1a = ConvBlock(in_channels, 16, down=True)
        self.enc1b = ConvBlock(16, 16, down=True)   # stacked CNNs

        self.enc2 = QBlock(16, 32, down=True)
        self.enc3 = QBlock(32, 64, down=True)
        self.enc4 = QBlock(64, 128, down=True)

        self.bottleneck = nn.Sequential(
            QuantumConv(128, 256),
            QuantumConv(256, 256)
        )

        # ----- Decoder -----
        self.up3 = QBlock(256, 128, down=False)     # concat with enc4
        self.up2 = QBlock(256, 64, down=False)      # concat with enc3
        self.up1 = QBlock(128, 32, down=False)      # concat with enc2

        self.up0a = ConvBlock(64, 32, down=False)   # concat with enc1
        self.up0b = ConvBlock(48, 32, down=False)   # handles 32+16=48

        # ----- Heads -----
        self.final_conv = nn.Conv2d(32, num_classes, 1)
        self.recon_conv = nn.Conv2d(32, in_channels, 1)
        
        # NEW auxiliary classifier for pseudo labels
        self.aux_classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1, x = self.enc1a(x)
        x1, x = self.enc1b(x)    # after 2nd convblock
        x2, x = self.enc2(x)
        x3, x = self.enc3(x)
        x4, x = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        # Decoder stage 3
        x = self.up3(x)
        if x.shape[2:] != x4.shape[2:]:
            x = F.interpolate(x, size=x4.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x4], dim=1)
        
        # Decoder stage 2
        x = self.up2(x)
        if x.shape[2:] != x3.shape[2:]:
            x = F.interpolate(x, size=x3.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x3], dim=1)
        
        # Decoder stage 1
        x = self.up1(x)
        if x.shape[2:] != x2.shape[2:]:
            x = F.interpolate(x, size=x2.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x2], dim=1)
        
        # Final stage
        x = self.up0a(x)
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x1], dim=1)

        x = self.up0b(x)

        # Heads
        seg_logits = self.final_conv(x)
        recon = self.recon_conv(x)
        
        # auxiliary classifier for pseudo-labels
        aux_logits = self.aux_classifier(recon)  # [B, num_classes, H, W]
        
        return seg_logits, recon, aux_logits
        
        
# -------------------------
# Model selector (common entry)
# -------------------------
def get_model(model_name, num_classes=4, in_channels=3, img_size=256):
    model_name = model_name.lower()

    if model_name == "unet":
        return UNet(num_classes=num_classes, in_channels=in_channels)

    elif model_name == "deeplabv3":
        return build_deeplabv3(num_classes=num_classes, in_channels=in_channels)

    elif model_name == "segformer":
        return SegFormer(num_classes=num_classes, in_channels=in_channels)
        
    elif model_name == "transunet":
        return TransUNet(num_classes=num_classes, in_channels=in_channels)
        
    elif model_name == "quantumunet":
        return QuantumUNet(num_classes=num_classes, in_channels=in_channels)

    else:
        raise ValueError(f"Unknown model name: {model_name}")


