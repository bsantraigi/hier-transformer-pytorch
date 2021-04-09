<img src="./HIER_Encoder-combined.png" height="400px"></img>

**Figure 1:** Detailed Architecture for a **Hierarchical Transformer Encoder** or **HT-Encoder**: The main inductive bias incorporated in this model is to encode the full dialog context hierarchically in two stages. This is done by the two encoders, 1) Shared Utterance Encoder (M layers) and 2) Context Encoder (N layers), as shown in the figure. Shared encoder first encodes each utterance (![formula](https://render.githubusercontent.com/render/math?math=u_1,%20u_2,%20\dots,%20u_t)) individually to extract the utterance level features. The same parameterized Shared Encoder is used for encoding all utterances in the context. In the second Context Encoder the full context is encoded using a single transformer encoder for extracting dialog level features. The attention mask in context encoder decides how the context encoding is done and is a choice of the user. This one depicted in the figure is for the HIER model described in Section 2.3 of paper. Only the final utterance in the Context Encoder gets to attend over all the previous utterances as shown. This allows the model to have access to both utterance level features and dialog level features till the last layer of the encoding process. Notation: Utterance ![formula](https://render.githubusercontent.com/render/math?math=i), ![formula](https://render.githubusercontent.com/render/math?math=u_i%20=%20[w_{i1},%20\dots,%20w_{i|u_i|}]), ![formula](https://render.githubusercontent.com/render/math?math=w_{ij}) is the word embedding for ![formula](https://render.githubusercontent.com/render/math?math=j^{th}) word in ![formula](https://render.githubusercontent.com/render/math?math=i^{th}) utterance.

# HIER - Pytorch

Implementation of <a href="https://arxiv.org/abs/2011.08067">HIER</a>, in Pytorch

> **Title**: Hierarchical Transformer for Task Oriented Dialog Systems.
> *Bishal Santra, Potnuru Anusha and Pawan Goyal* (NAACL 2021, Long Paper)



## Install

```bash

```

## Usage

```python
import torch
from hier_transformer_pytorch import HIERTransformer

hier_transformer = HIERTransformer(nhead=16, num_encoder_layers=12)
src = torch.rand((10, 32, 512))
tgt = torch.rand((20, 32, 512))
out = hier_transformer(src, tgt)
```

## Running the Experiments

```bash
Coming soon...
```

## Citations

```bibtex
@misc{santra2021hierarchical,
      title={Hierarchical Transformer for Task Oriented Dialog Systems}, 
      author={Bishal Santra and Potnuru Anusha and Pawan Goyal},
      year={2021},
      eprint={2011.08067},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgements

We thank the authors and developers 