---
tags:
- IPAdapter
---

# IPAdapter Batch (Adv.)
## Documentation
- Class name: `IPAdapterBatch`
- Category: `ipadapter`
- Output node: `False`

IPAdapterBatch is designed to enhance image processing capabilities by allowing batch processing of images with advanced image processing adapters. It extends the functionality of IPAdapterAdvanced by enabling the handling of multiple images simultaneously, optimizing the workflow for bulk image manipulation tasks.
## Input types
### Required
- **`model`**
    - Specifies the model to be used for image processing, serving as the core component for applying image transformations.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`ipadapter`**
    - Defines the image processing adapter to be applied to the images, dictating the specific transformations or enhancements to be performed.
    - Comfy dtype: `IPADAPTER`
    - Python dtype: `str`
- **`image`**
    - Represents the images to be processed, allowing for batch processing of multiple images in a single operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `list of str`
- **`weight`**
    - Determines the influence of the adapter's effect on the images, with the ability to adjust the intensity of the applied transformations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_type`**
    - Specifies the method for applying weights to the image transformations, influencing the overall effect of the adapter.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`start_at`**
    - Defines the starting point of the adapter's effect, allowing for fine-tuned control over the application of image enhancements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_at`**
    - Sets the endpoint of the adapter's effect, enabling precise control over the extent of image transformations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`embeds_scaling`**
    - Controls how embeddings are scaled during the processing, affecting the adaptation of the image features.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list of str`
- **`encode_batch_size`**
    - Specifies the batch size for encoding operations, optimizing the processing efficiency for large sets of images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`image_negative`**
    - Optional parameter for providing negative images to contrast with the primary images, enhancing the adapter's effect.
    - Comfy dtype: `IMAGE`
    - Python dtype: `list of str`
- **`attn_mask`**
    - Optional parameter for applying attention masks to the images, directing focus to specific areas during processing.
    - Comfy dtype: `MASK`
    - Python dtype: `list of str`
- **`clip_vision`**
    - Optional parameter for integrating CLIP vision features, enriching the image processing with additional contextual understanding.
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `str`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The processed model after applying the image processing adapter, reflecting the transformations or enhancements made to the images.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class IPAdapterBatch(IPAdapterAdvanced):
    def __init__(self):
        self.unfold_batch = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
                "encode_batch_size": ("INT", { "default": 0, "min": 0, "max": 4096 }),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }

```
