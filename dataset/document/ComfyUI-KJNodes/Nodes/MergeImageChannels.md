# Merge Image Channels
## Documentation
- Class name: `MergeImageChannels`
- Category: `KJNodes/image`
- Output node: `False`

The MergeImageChannels node is designed for combining separate image channels (red, green, and blue) into a single composite image. It optionally supports the inclusion of an alpha channel for transparency, allowing for more complex image compositions.
## Input types
### Required
- **`red`**
    - The red channel of the image, which contributes to the overall color composition by defining the red intensity.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`green`**
    - The green channel of the image, which contributes to the overall color composition by defining the green intensity.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`blue`**
    - The blue channel of the image, which contributes to the overall color composition by defining the blue intensity.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
### Optional
- **`mask`**
    - An optional alpha channel for adding transparency to the final image, enhancing its compositional complexity.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image after merging the red, green, and blue channels, optionally with an alpha channel for transparency.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MergeImageChannels:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "red": ("IMAGE",),
            "green": ("IMAGE",),
            "blue": ("IMAGE",),
            
            },
            "optional": {
                "mask": ("MASK", {"default": None}),
                },
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "merge"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Merges channel data into an image.  
"""
        
    def merge(self, red, green, blue, alpha=None):
        image = torch.stack([
        red[..., 0, None], # Red channel
        green[..., 1, None], # Green channel
        blue[..., 2, None]   # Blue channel
        ], dim=-1)
        image = image.squeeze(-2)
        if alpha is not None:
            image = torch.cat([image, alpha], dim=-1)
        return (image,)

```
