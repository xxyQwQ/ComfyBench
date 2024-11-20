---
tags:
- Mask
- MaskGeneration
---

# Create Gradient Mask
## Documentation
- Class name: `CreateGradientMask`
- Category: `KJNodes/masking/generate`
- Output node: `False`

This node is designed to generate a series of gradient masks based on specified dimensions and frame count. It allows for the creation of dynamic, time-offset gradients that can be inverted, offering flexibility in mask generation for various visual effects.
## Input types
### Required
- **`invert`**
    - Determines whether the generated gradient mask should be inverted. Inverting the mask swaps the gradient direction, enabling more creative control over the visual output.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`frames`**
    - Specifies the number of frames (or images) to generate, each with a slightly offset gradient to create a dynamic, time-evolving effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`width`**
    - Sets the width of the gradient mask, defining the horizontal dimension of the generated images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Sets the height of the gradient mask, defining the vertical dimension of the generated images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - Outputs a tensor representing the generated gradient mask(s), which can be used directly in image processing or visual effects pipelines.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CreateGradientMask:
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "createmask"
    CATEGORY = "KJNodes/masking/generate"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "invert": ("BOOLEAN", {"default": False}),
                 "frames": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
                 "width": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
                 "height": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
        },
    } 
    def createmask(self, frames, width, height, invert):
        # Define the number of images in the batch
        batch_size = frames
        out = []
        # Create an empty array to store the image batch
        image_batch = np.zeros((batch_size, height, width), dtype=np.float32)
        # Generate the black to white gradient for each image
        for i in range(batch_size):
            gradient = np.linspace(1.0, 0.0, width, dtype=np.float32)
            time = i / frames  # Calculate the time variable
            offset_gradient = gradient - time  # Offset the gradient values based on time
            image_batch[i] = offset_gradient.reshape(1, -1)
        output = torch.from_numpy(image_batch)
        mask = output
        out.append(mask)
        if invert:
            return (1.0 - torch.cat(out, dim=0),)
        return (torch.cat(out, dim=0),)

```
