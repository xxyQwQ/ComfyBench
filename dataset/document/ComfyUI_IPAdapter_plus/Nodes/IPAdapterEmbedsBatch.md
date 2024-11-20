---
tags:
- IPAdapter
---

# IPAdapter Embeds Batch
## Documentation
- Class name: `IPAdapterEmbedsBatch`
- Category: `ipadapter/embeds`
- Output node: `False`

The IPAdapterEmbedsBatch node is designed to handle batch processing of image embeddings within the IPAdapter framework. It extends the capabilities of IPAdapterEmbeds by enabling the processing of multiple embeddings simultaneously, optimizing for efficiency and scalability in embedding operations.
## Input types
### Required
- **`model`**
    - unknown
    - Comfy dtype: `MODEL`
    - Python dtype: `unknown`
- **`ipadapter`**
    - unknown
    - Comfy dtype: `IPADAPTER`
    - Python dtype: `unknown`
- **`pos_embed`**
    - unknown
    - Comfy dtype: `EMBEDS`
    - Python dtype: `unknown`
- **`weight`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`weight_type`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`start_at`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`end_at`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`embeds_scaling`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
### Optional
- **`neg_embed`**
    - unknown
    - Comfy dtype: `EMBEDS`
    - Python dtype: `unknown`
- **`attn_mask`**
    - unknown
    - Comfy dtype: `MASK`
    - Python dtype: `unknown`
- **`clip_vision`**
    - unknown
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `unknown`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - Represents the output model after processing the batch of image embeddings, encapsulating the enhanced or modified embeddings.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class IPAdapterEmbedsBatch(IPAdapterEmbeds):
    def __init__(self):
        self.unfold_batch = True

```
