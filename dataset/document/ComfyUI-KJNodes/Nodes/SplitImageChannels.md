# Split Image Channels
## Documentation
- Class name: `SplitImageChannels`
- Category: `KJNodes/image`
- Output node: `False`

The SplitImageChannels node is designed to separate the individual color channels of an image into distinct images, each channel being expanded across all channels of its respective output image, and additionally extracts the alpha channel as a separate mask. This process facilitates operations that require manipulation or analysis of specific color components within an image.
## Input types
### Required
- **`image`**
    - The input image to be split into its constituent color channels. This image is expected to have multiple channels, including an alpha channel for transparency.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`red`**
    - Comfy dtype: `IMAGE`
    - The red channel of the input image, expanded across all three color channels of the output image.
    - Python dtype: `torch.Tensor`
- **`green`**
    - Comfy dtype: `IMAGE`
    - The green channel of the input image, expanded across all three color channels of the output image.
    - Python dtype: `torch.Tensor`
- **`blue`**
    - Comfy dtype: `IMAGE`
    - The blue channel of the input image, expanded across all three color channels of the output image.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The alpha channel of the input image, representing the transparency mask.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SplitImageChannels:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "image": ("IMAGE",),
            },
            }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("red", "green", "blue", "mask")
    FUNCTION = "split"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Splits image channels into images where the selected channel  
is repeated for all channels, and the alpha as a mask. 
"""
        
    def split(self, image):
        red = image[:, :, :, 0:1] # Red channel
        green = image[:, :, :, 1:2] # Green channel
        blue = image[:, :, :, 2:3] # Blue channel
        alpha = image[:, :, :, 3:4] # Alpha channel
        alpha = alpha.squeeze(-1)

        # Repeat the selected channel for all channels
        red = torch.cat([red, red, red], dim=3)
        green = torch.cat([green, green, green], dim=3)
        blue = torch.cat([blue, blue, blue], dim=3)
        return (red, green, blue, alpha)

```
