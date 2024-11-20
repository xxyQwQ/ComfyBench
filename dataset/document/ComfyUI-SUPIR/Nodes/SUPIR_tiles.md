---
tags:
- Image
- Tiled
---

# SUPIR Tiles Preview
## Documentation
- Class name: `SUPIR_tiles`
- Category: `SUPIR`
- Output node: `False`

This node is designed for generating a preview of image tiles, facilitating the visualization of how an image is segmented into smaller, manageable pieces for processing or analysis. It primarily serves in the context of image processing pipelines where handling images in tiles is advantageous for performance or algorithmic reasons.
## Input types
### Required
- **`image`**
    - The input image to be tiled. This parameter is crucial for determining how the image will be segmented into tiles, affecting the visualization output.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`tile_size`**
    - Specifies the dimensions of each tile. This parameter directly influences the granularity of the tiling process, impacting both the number of tiles generated and their respective sizes.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`tile_stride`**
    - Defines the stride for tiling the image. This parameter determines the overlap between tiles, affecting the total number of tiles generated and their arrangement.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image_tiles`**
    - Comfy dtype: `IMAGE`
    - A collection of image tiles derived from the input image. This output is significant for visualizing the segmentation and for further processing or analysis of individual tiles.
    - Python dtype: `torch.Tensor`
- **`tile_size`**
    - Comfy dtype: `INT`
    - The dimension of each tile, as specified in the input. This output is crucial for understanding the scale at which the image was segmented.
    - Python dtype: `int`
- **`tile_stride`**
    - Comfy dtype: `INT`
    - The stride used for tiling the image, as specified in the input. This output provides insight into the overlap and arrangement of the generated tiles.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SUPIR_tiles:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "tile_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
            "tile_stride": ("INT", {"default": 256, "min": 64, "max": 8192, "step": 64}),
          
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("image_tiles", "tile_size", "tile_stride",)
    FUNCTION = "tile"
    CATEGORY = "SUPIR"
    DESCRIPTION = """
Tiles the image with same function as the Tiled samplers use.  
Useful for previewing the tiling and generating captions per tile (WIP feature)
"""

    def tile(self, image, tile_size, tile_stride):

        def _sliding_windows(h: int, w: int, tile_size: int, tile_stride: int):
            hi_list = list(range(0, h - tile_size + 1, tile_stride))
            if (h - tile_size) % tile_stride != 0:
                hi_list.append(h - tile_size)

            wi_list = list(range(0, w - tile_size + 1, tile_stride))
            if (w - tile_size) % tile_stride != 0:
                wi_list.append(w - tile_size)

            coords = []
            for hi in hi_list:
                for wi in wi_list:
                    coords.append((hi, hi + tile_size, wi, wi + tile_size))
            return coords

        image = image.permute(0, 3, 1, 2)
        _, _, h, w = image.shape

        tiles_iterator = _sliding_windows(h, w, tile_size, tile_stride)

        tiles = []
        for hi, hi_end, wi, wi_end in tiles_iterator:
            tile = image[:, :, hi:hi_end, wi:wi_end]
            
            tiles.append(tile)
        out = torch.cat(tiles, dim=0).to(torch.float32).permute(0, 2, 3, 1)
        print(out.shape)
        print("len(tiles): ", len(tiles))
        
        return (out, tile_size, tile_stride,)

```
