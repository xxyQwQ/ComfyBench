# EmptyLatentAudio
## Documentation
- Class name: `EmptyLatentAudio`
- Category: `_for_testing/audio`
- Output node: `False`

The EmptyLatentAudio node is designed to generate a blank audio latent space representation. It creates a tensor of zeros that serves as a placeholder or starting point for further audio processing or generation tasks.
## Input types
### Required
- **`seconds`**
    - Specifies the duration in seconds for which the blank audio latent space is to be generated. This duration determines the size of the resulting tensor.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - Outputs a tensor representing a blank audio latent space, which can be used as a starting point for audio generation or processing.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class EmptyLatentAudio:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"seconds": ("FLOAT", {"default": 47.6, "min": 1.0, "max": 1000.0, "step": 0.1})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "_for_testing/audio"

    def generate(self, seconds):
        batch_size = 1
        length = round((seconds * 44100 / 2048) / 2) * 2
        latent = torch.zeros([batch_size, 64, length], device=self.device)
        return ({"samples":latent, "type": "audio"}, )

```
