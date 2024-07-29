## StateFormer - Pytorch

Mamba-TTS Encoder

## Usage

```python
import torch

from StateFormer.stateformer import StateFormer
#Encoder only

model = StateFormer(n_mels=400, d_model=768, n_layer=12)
audio = torch.randn(1, 400, 128)
out = model(audio)


#Seq2Seq
from StateFormer.stateformer import StateFormerSeq2Seq

model = StateFormerSeq2Seq(n_mels = 400, d_model=768, heads=12, n_vocab = 51065, encoder_n_layers =12, decoder_n_layer=12)
audio = torch.randn(1, 400, 128)
text = torch.randint(0, 10000, (1, 128))
out = model(audio, text)

```

