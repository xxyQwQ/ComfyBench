---
tags:
- IPAdapter
---

# IPAdapter Precise Style Transfer Batch
## Documentation
- Class name: `IPAdapterPreciseStyleTransferBatch`
- Category: `ipadapter`
- Output node: `False`

This node specializes in applying precise style transfer techniques to images in batch mode, leveraging advanced image processing algorithms to blend styles from one image to another with high fidelity. It extends the capabilities of IPAdapterPreciseStyleTransfer by handling multiple images simultaneously, optimizing for efficiency and scalability in style transfer operations.
## Input types
### Required
- **`model`**
    - Specifies the model to be used for the style transfer process, serving as the core computational engine.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`ipadapter`**
    - Refers to the IPAdapter instance used for integrating various image processing functionalities and models.
    - Comfy dtype: `IPADAPTER`
    - Python dtype: `str`
- **`image`**
    - The target image(s) to which the style will be transferred, serving as the canvas for the style application.
    - Comfy dtype: `IMAGE`
    - Python dtype: `str`
- **`weight`**
    - Controls the intensity of the style transfer effect, allowing for fine-tuning of the style application's strength.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`style_boost`**
    - Adjusts the intensity of the style enhancement, enabling more pronounced style features in the transferred style.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`combine_embeds`**
    - Determines the method for combining embeddings from different sources, affecting the final style transfer outcome.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`start_at`**
    - Defines the starting point in the model's layers for applying the style transfer, allowing for layer-specific style applications.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_at`**
    - Sets the endpoint in the model's layers for the style transfer, enabling precise control over the depth of style integration.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`embeds_scaling`**
    - Specifies the scaling method for embeddings, influencing how style features are emphasized or balanced.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`image_negative`**
    - An optional negative image to contrast with the primary style transfer, used for refining the style application.
    - Comfy dtype: `IMAGE`
    - Python dtype: `str`
- **`attn_mask`**
    - An optional attention mask to focus or exclude specific areas from the style transfer process.
    - Comfy dtype: `MASK`
    - Python dtype: `str`
- **`clip_vision`**
    - Optionally integrates CLIP vision models to guide the style transfer with semantic understanding.
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `str`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified model after applying the style transfer, showcasing the integration of the desired style into the target images.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class IPAdapterPreciseStyleTransferBatch(IPAdapterPreciseStyleTransfer):
    def __init__(self):
        self.unfold_batch = True

```
