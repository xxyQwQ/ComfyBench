---
tags:
- VisualEffects
---

# Calculate Normals From Images
## Documentation
- Class name: `CalculateNormalsFromImages`
- Category: `IC-Light`
- Output node: `False`

The CalculateNormalsFromImages node is designed to compute normal maps from a set of images representing different directional exposures. It processes batches of images to generate corresponding normal maps and optionally applies masks to refine the output, facilitating advanced image processing tasks such as lighting simulation or 3D reconstruction.
## Input types
### Required
- **`images`**
    - A batch of images, expected to be in sets of four representing left, right, bottom, and top directional exposures. These images are used to calculate the normal maps.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`sigma`**
    - A float value used to adjust the sharpness of the generated normal map. Higher values result in sharper details.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`center_input_range`**
    - A boolean flag indicating whether the input images should be normalized to a 0.5 center point before processing.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`mask`**
    - An optional mask to apply to the images, used to isolate or highlight specific areas for normal map calculation.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`normal`**
    - Comfy dtype: `IMAGE`
    - The generated normal map, representing the surface normals derived from the input images.
    - Python dtype: `torch.Tensor`
- **`divided`**
    - Comfy dtype: `IMAGE`
    - An intermediate representation of the processed images, used internally for normalization and calculation.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CalculateNormalsFromImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "sigma": ("FLOAT", { "default": 10.0, "min": 0.01, "max": 100.0, "step": 0.01, }),
                "center_input_range": ("BOOLEAN", { "default": False, }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("normal", "divided",)
    FUNCTION = "execute"
    CATEGORY = "IC-Light"
    DESCRIPTION = """
Calculates normal map from different directional exposures.  
Takes in 4 images as a batch:  
left, right, bottom, top  

"""

    def execute(self, images, sigma, center_input_range, mask=None):
        B, H, W, C = images.shape
        repetitions = B // 4        

        if center_input_range:
            images = images * 0.5 + 0.5
        if mask is not None:
            if mask.shape[-2:] != images[0].shape[:-1]:
                mask = mask.unsqueeze(0)
                mask = F.interpolate(mask, size=(images.shape[1], images.shape[2]), mode="bilinear")
                mask = mask.squeeze(0)
        

        
        normal_list = []
        divided_list = []
        iteration_counter = 0

        for i in range(0, B, 4):  # Loop over every 4 images
            index = torch.arange(iteration_counter, B, repetitions)
            rearranged_images = images[index]
            images_np = rearranged_images.numpy().astype(np.float32)
            
            left = images_np[0]
            right = images_np[1]
            bottom = images_np[2]
            top = images_np[3]

            ambient = (left + right + bottom + top) / 4.0
        
            def safe_divide(a, b):
                e = 1e-5
                return ((a + e) / (b + e)) - 1.0

            left = safe_divide(left, ambient)
            right = safe_divide(right, ambient)
            bottom = safe_divide(bottom, ambient)
            top = safe_divide(top, ambient)

            u = (right - left) * 0.5
            v = (top - bottom) * 0.5

            u = np.mean(u, axis=2)
            v = np.mean(v, axis=2)
            h = (1.0 - u ** 2.0 - v ** 2.0).clip(0, 1e5) ** (0.5 * sigma)
            z = np.zeros_like(h)

            normal = np.stack([u, v, h], axis=2)
            normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
            if mask is not None:
                matting = mask[iteration_counter].unsqueeze(0).numpy().astype(np.float32)
                matting = matting[..., np.newaxis]
                normal = normal * matting + np.stack([z, z, 1 - z], axis=2) 
                normal = torch.from_numpy(normal)
                #normal = normal.unsqueeze(0)
            else:
                normal = normal + np.stack([z, z, 1 - z], axis=2)
                normal = torch.from_numpy(normal).unsqueeze(0)

            iteration_counter += 1 
            normal = (normal - normal.min()) / ((normal.max() - normal.min()))
            normal_list.append(normal)
            divided = np.stack([left, right, bottom, top])
            divided = torch.from_numpy(divided)
            divided = (divided - divided.min()) / ((divided.max() - divided.min()))
            divided = torch.max(divided, dim=3, keepdim=True)[0].repeat(1, 1, 1, 3)
            divided_list.append(divided)

        normal_out = torch.cat(normal_list, dim=0)
        divided_out = torch.cat(divided_list, dim=0)
   
        return (normal_out, divided_out, )

```
