import torch
from hier_transformer_pytorch import HIERTransformer
import hier_transformer_pytorch as HIER

hier_transformer = HIERTransformer(nhead=16, num_encoder_layers=12)
src = torch.rand((10, 32, 512))
tgt = torch.rand((20, 32, 512))
out = hier_transformer(src, tgt)

print(f"src: {src.shape}, tgt: {tgt.shape} -> out: {out.shape}")
# print(out)

"""CT Masks
attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length.
"""
import seaborn as sns
import matplotlib.pyplot as plt

src = torch.rand((32, 10))
tgt = torch.rand((32, 20))
src_padding_mask = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1]).unsqueeze(0).expand(32, -1)
src_attn_mask = torch.rand(32, 20, 10) > 0.5
utt_indices = torch.tensor([0, 0, 1, 1, 1, 2, 2, -1, -1, -1]).unsqueeze(0).expand(32, -1)

pe_utt_loc, enc_mask_utt, enc_mask_ct, dec_enc_attn_mask = HIER.hier_masks._CLS_masks(tgt, src, src_padding_mask, utt_indices)

# Uncomment this for plotting masks
print(src_padding_mask[0])
print("===============>")
print(utt_indices[0])
print((1 - src_padding_mask[0] * 1.) * utt_indices[0])

sns.set_style("whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Various HIER-CLS Masks')

sns.heatmap((src_padding_mask[0]).unsqueeze(0).expand(tgt.shape[1], -1).cpu().numpy(), ax=axes[0,0]).set_title("SRC Padding Mask")
sns.heatmap((enc_mask_utt[0] * 1).cpu().numpy(), ax=axes[0,1]).set_title("UT_Mask")
sns.heatmap((enc_mask_ct[0] * 1).cpu().numpy(), ax=axes[1,0]).set_title("CT_Mask")
sns.heatmap((dec_enc_attn_mask[0] * 1).cpu().numpy(), ax=axes[1,1]).set_title("Dec_2_Enc_Mask")
plt.show()
