---
tags:
- DepthMap
- DepthMapEstimation
- Image
---

# Colorize Depthmap
## Documentation
- Class name: `ColorizeDepthmap`
- Category: `Marigold`
- Output node: `False`

The ColorizeDepthmap node is designed to transform depth maps into colorized representations, enhancing visual interpretation by applying a colormap. This process facilitates easier understanding and analysis of depth information by converting grayscale depth maps into vibrant, color-coded images.
## Input types
### Required
- **`image`**
    - The depth map to be colorized, which can be either a torch.Tensor or a numpy.ndarray. This map represents the depth information of a scene as a 2D array.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Union[torch.Tensor, numpy.ndarray]`
- **`colorize_method`**
    - The name of the matplotlib colormap to use for colorizing the depth map.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The colorized depth map, enhanced with a colormap for better visual interpretation. The output is in the same data type as the input depth map.
    - Python dtype: `Union[torch.Tensor, numpy.ndarray]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ColorizeDepthmap:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {  
            "image": ("IMAGE", ),
            "colorize_method": (
            [   
                'Spectral',
                'terrain', 
                'viridis',
                'plasma',
                'inferno',
                'magma',
                'cividis',
                'twilight',
                'rainbow',
                'gist_rainbow',
                'gist_ncar',
                'gist_earth',
                'turbo',
                'jet',
                'afmhot',
                'copper',
                'seismic',
                'hsv',
                'brg',

            ], {
               "default": 'Spectral'
            }),
            },
            
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("image",)
    FUNCTION = "color"

    CATEGORY = "Marigold"

    def color(self, image, colorize_method):
        colored_images = []
        for i in range(image.shape[0]):  # Iterate over the batch dimension
            depth_map = image[i].squeeze().permute(2, 0, 1)
            depth_map = depth_map[0]
            depth_map = colorizedepth(depth_map, colorize_method)
            depth_map = torch.from_numpy(depth_map) / 255
            depth_map = depth_map.unsqueeze(0)
            colored_images.append(depth_map)
        
        # Stack the list of tensors along a new dimension
        colored_images = torch.cat(colored_images, dim=0)
        return (colored_images,)

```
