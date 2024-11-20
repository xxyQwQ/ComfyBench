# EmptySD3LatentImage
## Documentation
- Class name: `EmptySD3LatentImage`
- Category: `latent/sd3`
- Output node: `False`

This node is designed to generate a blank latent image with a specific shape and initial value for use in SD3 models. It primarily serves as a starting point for further processing or manipulation within the SD3 framework.
## Input types
### Required
- **`width`**
    - Specifies the width of the latent image to be generated. It determines the horizontal dimension of the output latent tensor.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Specifies the height of the latent image to be generated. It affects the vertical dimension of the output latent tensor.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`batch_size`**
    - Determines the number of latent images to generate in a single batch. This allows for the creation of multiple latent images simultaneously.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The output is a latent representation in the form of a tensor, initialized with a specific value, ready for further processing in the SD3 pipeline.
    - Python dtype: `Tuple[Dict[str, torch.Tensor]]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class EmptySD3LatentImage:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent/sd3"

    def generate(self, width, height, batch_size=1):
        latent = torch.ones([batch_size, 16, height // 8, width // 8], device=self.device) * 0.0609
        return ({"samples":latent}, )

```
