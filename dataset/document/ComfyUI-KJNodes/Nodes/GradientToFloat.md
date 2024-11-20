# Gradient To Float
## Documentation
- Class name: `GradientToFloat`
- Category: `KJNodes/image`
- Output node: `False`

Transforms an image into a list of float values based on sampling along its width and height axes. This process is useful for extracting numerical representations of images for further analysis or processing.
## Input types
### Required
- **`image`**
    - The image tensor to be sampled, where the sampling occurs along the width and height axes to generate float values.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`steps`**
    - Defines the number of intervals to sample along the width and height axes of the image, affecting the resolution of the output float lists.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`float_x`**
    - Comfy dtype: `FLOAT`
    - A list of float values representing the mean of the sampled points along the width of the image.
    - Python dtype: `List[float]`
- **`float_y`**
    - Comfy dtype: `FLOAT`
    - A list of float values representing the mean of the sampled points along the height of the image.
    - Python dtype: `List[float]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class GradientToFloat:
    
    RETURN_TYPES = ("FLOAT", "FLOAT",)
    RETURN_NAMES = ("float_x", "float_y", )
    FUNCTION = "sample"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Calculates list of floats from image.    
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "steps": ("INT", {"default": 10, "min": 2, "max": 10000, "step": 1}),
        },
    } 
    
    def sample(self, image, steps):
        # Assuming image is a tensor with shape [B, H, W, C]
        B, H, W, C = image.shape

        # Sample along the width axis (W)
        w_intervals = torch.linspace(0, W - 1, steps=steps, dtype=torch.int64)
        # Assuming we're sampling from the first batch and the first channel
        w_sampled = image[0, :, w_intervals, 0]

        # Sample along the height axis (H)
        h_intervals = torch.linspace(0, H - 1, steps=steps, dtype=torch.int64)
        # Assuming we're sampling from the first batch and the first channel
        h_sampled = image[0, h_intervals, :, 0]

        # Taking the mean across the height for width sampling, and across the width for height sampling
        w_values = w_sampled.mean(dim=0).tolist()
        h_values = h_sampled.mean(dim=1).tolist()

        return (w_values, h_values)

```
