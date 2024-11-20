# Frequency Combine
## Documentation
- Class name: `FrequencyCombine`
- Category: `image/filters`
- Output node: `False`

The FrequencyCombine node is designed to blend high and low-frequency image components using specified mathematical operations. It enables the enhancement or modification of images by combining frequency elements in unique ways, offering a method to manipulate image textures and details.
## Input types
### Required
- **`high_frequency`**
    - The high-frequency component of an image, representing details such as edges and textures. It plays a crucial role in defining the sharpness and detail of the combined output.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`low_frequency`**
    - The low-frequency component of an image, encapsulating broad features and color gradients. It forms the base upon which high-frequency details are added or modified.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mode`**
    - Specifies the mathematical operation ('subtract' or 'divide') to combine high and low-frequency components, affecting the texture and detail enhancement of the output image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`eps`**
    - A small epsilon value used in the 'divide' mode to prevent division by zero, ensuring numerical stability.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image after combining high and low-frequency components, showcasing enhanced or altered textures and details.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class FrequencyCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "high_frequency": ("IMAGE", ),
                "low_frequency": ("IMAGE", ),
                "mode": (["subtract", "divide"],),
                "eps": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.99, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "filter_image"

    CATEGORY = "image/filters"

    def filter_image(self, high_frequency, low_frequency, mode, eps):
        t = low_frequency.detach().clone()
        if mode == "subtract":
            t = t + high_frequency - 0.5
        else:
            t = (high_frequency * 2) * (t + eps) - eps
        return (torch.clamp(t, 0, 1),)

```
