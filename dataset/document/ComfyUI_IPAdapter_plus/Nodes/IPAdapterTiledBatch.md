---
tags:
- IPAdapter
---

# IPAdapter Tiled Batch
## Documentation
- Class name: `IPAdapterTiledBatch`
- Category: `ipadapter/tiled`
- Output node: `False`

The IPAdapterTiledBatch node is designed to apply image processing adaptations in a batched manner to tiled images, leveraging the capabilities of an underlying IPAdapterTiled class. It focuses on enhancing or modifying images through a series of transformations, with an emphasis on batch processing for efficiency and scalability.
## Input types
### Required
- **`model`**
    - Specifies the model to which the image processing adaptations will be applied. It is central to the node's operation, determining the transformations that will be performed on the input images.
    - Comfy dtype: `MODEL`
    - Python dtype: `MODEL`
- **`ipadapter`**
    - Defines the specific IPAdapter to be used for the image processing adaptations. This parameter is crucial for determining the nature of the transformations applied to the images.
    - Comfy dtype: `IPADAPTER`
    - Python dtype: `IPADAPTER`
- **`image`**
    - The input image to be processed. This parameter is essential for the node's functionality, serving as the primary data on which the adaptations are applied.
    - Comfy dtype: `IMAGE`
    - Python dtype: `IMAGE`
- **`weight`**
    - Affects the intensity of the applied adaptations, allowing for fine-tuning of the transformation's impact on the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `FLOAT`
- **`weight_type`**
    - Determines the method of weighting the adaptations, influencing how the transformations are applied based on the specified type.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `WEIGHT_TYPES`
- **`start_at`**
    - Defines the starting point of the adaptation process, enabling control over when the transformations begin to take effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `FLOAT`
- **`end_at`**
    - Specifies the endpoint of the adaptation process, allowing for precise control over the duration and extent of the transformations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `FLOAT`
- **`sharpening`**
    - Controls the level of image sharpening applied during the adaptation process, enhancing the clarity and detail of the transformed images.
    - Comfy dtype: `FLOAT`
    - Python dtype: `FLOAT`
- **`embeds_scaling`**
    - Determines how the embeddings are scaled during the adaptation process, affecting the overall impact of the transformations on the image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`encode_batch_size`**
    - Specifies the batch size for encoding operations, optimizing the efficiency of the adaptation process by adjusting the volume of data processed at once.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`image_negative`**
    - An optional negative image input that can be used for contrastive adaptations, enhancing the effectiveness of certain transformations.
    - Comfy dtype: `IMAGE`
    - Python dtype: `IMAGE`
- **`attn_mask`**
    - An optional attention mask that can be applied to focus or restrict the adaptations to specific areas of the image.
    - Comfy dtype: `MASK`
    - Python dtype: `MASK`
- **`clip_vision`**
    - An optional parameter that, when provided, integrates CLIP vision features into the adaptation process, enriching the transformations with additional visual context.
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `CLIP_VISION`
## Output types
- **`MODEL`**
    - Comfy dtype: `MODEL`
    - The model after the image processing adaptations have been applied, reflecting the changes made to the input images.
    - Python dtype: `MODEL`
- **`tiles`**
    - Comfy dtype: `IMAGE`
    - The processed tiles of the image, showcasing the results of the applied adaptations.
    - Python dtype: `List[IMAGE]`
- **`masks`**
    - Comfy dtype: `MASK`
    - The masks used or generated during the adaptation process, indicating areas of focus or alteration.
    - Python dtype: `List[MASK]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class IPAdapterTiledBatch(IPAdapterTiled):
    def __init__(self):
        self.unfold_batch = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "sharpening": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05 }),
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
