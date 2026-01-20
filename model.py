import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from transformers import SegformerModel, SegformerConfig


class SRMConv(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.srm_layer = nn.Conv2d(in_channels, 3, 5, 1, 2, bias=False)
        self._initialize_srm_weights()
        for param in self.srm_layer.parameters():
            param.requires_grad = False 

    def _initialize_srm_weights(self):
        q = [4.0, 12.0, 2.0]
        filter1 = [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]]
        filter2 = [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]
        filter3 = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        
        weight = torch.tensor(np.stack([np.array(f)/q_val for f, q_val in zip([filter1, filter2, filter3], q)]), dtype=torch.float32).unsqueeze(1)
        self.srm_layer.weight.data = weight.repeat(1, 3, 1, 1) / 3.0

    def forward(self, x):
        return self.srm_layer(x)

class ZeroWindow:
    def __init__(self): self.store = {}
    def __call__(self, x_in, h, w, rat_s=0.1):
        sigma = h * rat_s, w * rat_s
        key = str(x_in.shape) + str(rat_s) + str(x_in.device)
        if key not in self.store:
            device = x_in.device
            b, c, h2, w2 = x_in.shape
            ind_r = torch.arange(h2, device=device).float().view(1, 1, -1, 1).expand_as(x_in)
            ind_c = torch.arange(w2, device=device).float().view(1, 1, 1, -1).expand_as(x_in)
            c_indices = torch.from_numpy(np.indices((h, w))).float().to(device)
            cent_r = c_indices[0].reshape(-1).reshape(1, c, 1, 1).expand_as(x_in)
            cent_c = c_indices[1].reshape(-1).reshape(1, c, 1, 1).expand_as(x_in)
            gaus = torch.exp(-(ind_r - cent_r)**2 / (2 * sigma[0]**2)) * torch.exp(-(ind_c - cent_c)**2 / (2 * sigma[1]**2))
            self.store[key] = (1 - gaus)
        return self.store[key] * x_in

class Corr(nn.Module):
    def __init__(self, topk=3, max_resolution=128):
        super().__init__()
        self.topk = topk
        self.max_resolution = max_resolution
        self.zero_window = ZeroWindow()
        self.alpha = nn.Parameter(torch.tensor(5., dtype=torch.float32))

    def forward(self, x):
        b, c, h, w = x.shape
        

        is_downsampled = False
        if h > self.max_resolution or w > self.max_resolution:
            is_downsampled = True
            orig_h, orig_w = h, w

            x_in = F.adaptive_avg_pool2d(x, (self.max_resolution, self.max_resolution))
            h1, w1 = self.max_resolution, self.max_resolution
        else:
            x_in = x
            h1, w1 = h, w


        xn = F.normalize(x_in, p=2, dim=-3)

        x_aff = torch.matmul(xn.permute(0, 2, 3, 1).reshape(b, -1, c), xn.reshape(b, c, -1))
        
        x_aff = self.zero_window(x_aff.reshape(b, -1, h1, w1), h1, w1, rat_s=0.05).reshape(b, h1 * w1, h1 * w1)
        
        alpha = self.alpha.to(x.device)
        x_c = F.softmax(x_aff * alpha, dim=-1) * F.softmax(x_aff * alpha, dim=-2)
        

        val, _ = torch.topk(x_c.reshape(b, h1 * w1, h1, w1), k=self.topk, dim=-3)

        if is_downsampled:

            val = F.interpolate(val, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            
        return val

class TransformerBottleneck(nn.Module):
    def __init__(self, in_channels, num_heads=8, mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.layernorm2 = nn.LayerNorm(in_channels)
        self.mlp = nn.Sequential(nn.Linear(in_channels, mlp_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(mlp_dim, in_channels), nn.Dropout(dropout))
    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = x_flat + self.attn(self.layernorm1(x_flat), self.layernorm1(x_flat), self.layernorm1(x_flat))[0]
        x_flat = x_flat + self.mlp(self.layernorm2(x_flat))
        return x_flat.transpose(1, 2).reshape(b, c, h, w)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
    def forward(self, x):
        avg, mx = torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]
        return x + x * torch.sigmoid(self.conv1(torch.cat([avg, mx], dim=1)))

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)
        )
    def forward(self, x): return x + self.conv(x) if self.use_res_connect else self.conv(x)



class InstanceHead(nn.Module):
    def __init__(self, in_channels, max_instances=5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.attn = TransformerBottleneck(128, num_heads=4, mlp_dim=256)

        self.conv2 = nn.Conv2d(128, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.final = nn.Conv2d(64, max_instances, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        

        x_small = F.adaptive_avg_pool2d(x, (64, 64)) 
        x_small = self.attn(x_small)

        x_global = F.interpolate(x_small, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + x_global
        
        x = self.relu(self.bn2(self.conv2(x)))
        return self.final(x)


class UnetMitB51(nn.Module):
    def __init__(self, num_instances=5, pretrained=True, device1='cuda:0', device2='cuda:1', device3='cuda:2', model_path="nvidia/mit-b5"):
        super().__init__()
        
        self.dev1 = torch.device(device1)
        self.dev2 = torch.device(device2)
        self.dev3 = torch.device(device3)
        


        self.srm = SRMConv(in_channels=3).to(self.dev1)


        if pretrained:
            print(f"Loading weights from: {model_path}")
            self.backbone = SegformerModel.from_pretrained(model_path)
        else:
            config = SegformerConfig.from_pretrained(model_path)
            self.backbone = SegformerModel(config)
            
        self.backbone.to(self.dev1)

        old_proj = self.backbone.encoder.patch_embeddings[0].proj
        new_proj = nn.Conv2d(6, old_proj.out_channels, 
                             kernel_size=old_proj.kernel_size, 
                             stride=(2, 2), 
                             padding=old_proj.padding)
        new_proj.to(self.dev1)
        
        with torch.no_grad():
            new_proj.weight[:, :3] = old_proj.weight
            new_proj.weight[:, 3:] = old_proj.weight.mean(dim=1, keepdim=True)
            if old_proj.bias is not None: new_proj.bias = old_proj.bias
        self.backbone.encoder.patch_embeddings[0].proj = new_proj


        self.backbone.encoder.patch_embeddings[2].to(self.dev2)
        self.backbone.encoder.block[2].to(self.dev2)
        self.backbone.encoder.layer_norm[2].to(self.dev2)

        self.backbone.encoder.patch_embeddings[3].to(self.dev3)
        self.backbone.encoder.block[3].to(self.dev3)
        self.backbone.encoder.layer_norm[3].to(self.dev3)

        

        self.corr_x1 = Corr(topk=48).to(self.dev1)
        self.aspp_x1 = models.segmentation.deeplabv3.ASPP(48, [4, 8, 12, 16], 48).to(self.dev1)
        self.sam_x1 = SpatialAttention().to(self.dev1)


        self.corr_x2 = Corr(topk=64).to(self.dev3)
        self.aspp_x2 = models.segmentation.deeplabv3.ASPP(64, [4, 8, 12, 16], 64).to(self.dev3)
        self.sam_x2 = SpatialAttention().to(self.dev3)

        self.corr_x3 = Corr(topk=160).to(self.dev3)
        self.aspp_x3 = models.segmentation.deeplabv3.ASPP(160, [4, 8, 12, 16], 160).to(self.dev3)
        self.sam_x3 = SpatialAttention().to(self.dev3)

        # --- GPU 3: Decoder Assembly ---
        self.bottleneck_conv = nn.Conv2d(512, 512, 1).to(self.dev3)
        self.transformer = TransformerBottleneck(in_channels=512, dropout=0.1).to(self.dev3)
        
        self.dconv1 = nn.ConvTranspose2d(512, 160, 4, padding=1, stride=2).to(self.dev3)
        self.invres1 = InvertedResidual(160 + 160, 160, 1, 1).to(self.dev3)
        
        self.dconv2 = nn.ConvTranspose2d(160, 64, 4, padding=1, stride=2).to(self.dev3)
        self.invres2 = InvertedResidual(64 + 64, 64, 1, 1).to(self.dev3)
        
        self.dconv3 = nn.ConvTranspose2d(64, 48, 4, padding=1, stride=2).to(self.dev3)
        self.invres3 = InvertedResidual(48 + 48, 48, 1, 1).to(self.dev3)
        
        self.dconv4 = nn.ConvTranspose2d(48, 32, 4, padding=1, stride=2).to(self.dev3)
        self.invres4 = InvertedResidual(32, 32, 1, 1).to(self.dev3)

        self.instance_head = InstanceHead(in_channels=32, max_instances=num_instances).to(self.dev3)

    def forward(self, x):

        x = x.to(self.dev1).float()
        srm_x = self.srm(x) 
        x = torch.cat([x, srm_x], dim=1)
        
        out1, h, w = self.backbone.encoder.patch_embeddings[0](x)
        for block in self.backbone.encoder.block[0]:
            out1 = block(out1, h, w)[0]
        out1 = self.backbone.encoder.layer_norm[0](out1)
        
        B, N, C = out1.shape
        x1 = out1.transpose(1, 2).reshape(B, C, h, w) 

        out2, h, w = self.backbone.encoder.patch_embeddings[1](x1)
        

        for block in self.backbone.encoder.block[1]:
            out2 = torch.utils.checkpoint.checkpoint(
                lambda t, b=block, h=h, w=w: b(t, h, w)[0], 
                out2, 
                use_reentrant=False
            )
        out2 = self.backbone.encoder.layer_norm[1](out2)
        
        B, N, C = out2.shape
        x2 = out2.transpose(1, 2).reshape(B, C, h, w)


        x2_dev2 = x2.to(self.dev2) 
        out3, h, w = self.backbone.encoder.patch_embeddings[2](x2_dev2)

        for block in self.backbone.encoder.block[2]:
            out3 = torch.utils.checkpoint.checkpoint(
                lambda t, b=block, h=h, w=w: b(t, h, w)[0], 
                out3, 
                use_reentrant=False
            )
        out3 = self.backbone.encoder.layer_norm[2](out3)
        
        B, N, C = out3.shape
        x3 = out3.transpose(1, 2).reshape(B, C, h, w)

        x3_dev3 = x3.to(self.dev3) 
        out4, h, w = self.backbone.encoder.patch_embeddings[3](x3_dev3)

        for block in self.backbone.encoder.block[3]:
            out4 = torch.utils.checkpoint.checkpoint(
                lambda t, b=block, h=h, w=w: b(t, h, w)[0], 
                out4, 
                use_reentrant=False
            )
            
        out4 = self.backbone.encoder.layer_norm[3](out4)
        
        B, N, C = out4.shape
        x4 = out4.transpose(1, 2).reshape(B, C, h, w)
        
        B, N, C = out4.shape
        x4 = out4.transpose(1, 2).reshape(B, C, h, w)


        x1_proc = self.sam_x1(self.aspp_x1(self.corr_x1(x1)))

 
        x2_dev3_skip = x2.to(self.dev3)
        x2_proc = self.sam_x2(self.aspp_x2(self.corr_x2(x2_dev3_skip)))
        
        x3_dev3_skip = x3.to(self.dev3)
        x3_proc = self.sam_x3(self.aspp_x3(self.corr_x3(x3_dev3_skip)))
        

        x4_b = x4 + self.transformer(self.bottleneck_conv(x4))


        up1 = self.invres1(torch.cat([x3_proc, self.dconv1(x4_b)], dim=1))
        

        up2 = self.invres2(torch.cat([x2_proc, self.dconv2(up1)], dim=1))
        

        up3 = self.invres3(torch.cat([x1_proc.to(self.dev3), self.dconv3(up2)], dim=1))
        

        up4 = self.invres4(self.dconv4(up3)) 
        
        logits = self.instance_head(up4)
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return logits, up4