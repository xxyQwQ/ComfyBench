# Background Scaler
## Documentation
- Class name: `BackgroundScaler`
- Category: `IC-Light`
- Output node: `False`

The BackgroundScaler node is designed to adjust the grayscale level of masked areas within an image, allowing for dynamic scaling based on specified parameters. It provides a method to selectively modify the appearance of background elements in images by applying a scale factor to the masked regions.
## Input types
### Required
- **`image`**
    - The input image to be processed. It serves as the primary canvas on which the grayscale scaling is applied to the masked areas.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask`**
    - A mask that specifies the areas within the image to be scaled. The mask determines which parts of the image will undergo the grayscale adjustment.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`scale`**
    - A scale factor that determines the intensity of the grayscale to be applied to the masked areas. It allows for fine-tuning the visual impact of the scaling on the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`invert`**
    - A boolean flag that, when set to True, inverts the mask, thereby applying the scaling to the unmasked areas instead.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image with the masked areas adjusted in grayscale according to the specified scale factor.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class BackgroundScaler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "scale": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.001}),
                "invert": ("BOOLEAN", { "default": False, }),
            }
        }

    CATEGORY = "IC-Light"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    DESCRIPTION = """
Sets the masked area color in grayscale range.  
"""

    def apply(self, image: torch.Tensor, mask: torch.Tensor, scale: float, invert: bool):

        # Validate inputs
        if not isinstance(image, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise ValueError("image and mask must be torch.Tensor types.")
        if image.ndim != 4 or mask.ndim not in [3, 4]:
            raise ValueError("image must be a 4D tensor, and mask must be a 3D or 4D tensor.")

        # Adjust mask dimensions if necessary
        if mask.ndim == 3:
            # [B, H, W] => [B, H, W, C=1]
            mask = mask.unsqueeze(-1)

        if invert:
            mask = 1 - mask
        image_out = image * mask + (1 - mask) * scale
        image_out = torch.clamp(image_out, 0, 1).cpu().float()
        
        return (image_out,)

```
