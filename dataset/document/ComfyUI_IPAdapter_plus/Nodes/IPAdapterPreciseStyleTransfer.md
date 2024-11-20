# IPAdapter Precise Style Transfer
## Documentation
- Class name: `IPAdapterPreciseStyleTransfer`
- Category: `ipadapter`
- Output node: `False`

The IPAdapterPreciseStyleTransfer node is designed for executing precise style transfer operations on images within a given model's framework. It leverages advanced image processing techniques to blend and transfer styles between images, offering fine-grained control over the style transfer process through a comprehensive set of parameters.
## Input types
### Required
- **`model`**
    - Specifies the model to be used for the style transfer operation, serving as the foundation for the transformation process.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`ipadapter`**
    - Identifies the IPAdapter instance to be utilized, acting as a crucial component in managing and facilitating the style transfer.
    - Comfy dtype: `IPADAPTER`
    - Python dtype: `str`
- **`image`**
    - The target image to which the style will be transferred, serving as the canvas for the operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `str`
- **`weight`**
    - Controls the intensity of the style transfer, allowing for adjustable influence of the style on the target image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`style_boost`**
    - Amplifies the effect of the style being transferred, providing an additional layer of control over the style's impact.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`combine_embeds`**
    - Determines the method for combining embeddings from different sources, influencing the final outcome of the style transfer.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`start_at`**
    - Defines the starting point of the style transfer effect within the model's processing layers, offering control over the depth of style application.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_at`**
    - Sets the endpoint for the style transfer effect, allowing for precision in how deeply the style is integrated into the target image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`embeds_scaling`**
    - Specifies the approach for scaling embeddings, affecting the balance between key and value components in the style transfer.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`image_negative`**
    - An optional negative image to contrast with the target image, potentially guiding the style transfer in a specific direction.
    - Comfy dtype: `IMAGE`
    - Python dtype: `str`
- **`attn_mask`**
    - An optional attention mask that can be applied to focus or restrict the style transfer to specific regions of the image.
    - Comfy dtype: `MASK`
    - Python dtype: `str`
- **`clip_vision`**
    - Optionally integrates CLIP vision models to enhance the style transfer with semantic understanding from textual or visual cues.
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `str`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - Returns the model after applying the style transfer, reflecting the modifications and enhancements made during the process.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class IPAdapterPreciseStyleTransfer(IPAdapterAdvanced):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "style_boost": ("FLOAT", { "default": 1.0, "min": -5, "max": 5, "step": 0.05 }),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }

```
