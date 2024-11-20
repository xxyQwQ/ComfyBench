# LayerMask: RemBgUltra
## Documentation
- Class name: `LayerMask: RemBgUltra`
- Category: `😺dzNodes/LayerMask`
- Output node: `False`

The `LayerMask: RemBgUltra` node specializes in removing backgrounds from images with an emphasis on high detail preservation. It leverages advanced techniques to ensure that even fine details like hair or fur edges are retained after background removal, making it ideal for tasks requiring precise foreground extraction.
## Input types
### Required
- **`image`**
    - The `image` parameter represents the collection of images from which backgrounds are to be removed. It plays a crucial role in the node's operation by serving as the primary data for background removal processes.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`detail_range`**
    - Specifies the range within which detail enhancement and preservation are applied, affecting the precision of the background removal around edges.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`black_point`**
    - Defines the black point threshold for adjusting the contrast of the mask, influencing the visibility of finer details.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`white_point`**
    - Sets the white point threshold for mask contrast adjustment, further refining the detail visibility in the extracted foreground.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`process_detail`**
    - A boolean flag that determines whether additional detail processing is applied, enhancing the edge definition and overall detail preservation in the mask.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - A tensor containing the images with their backgrounds removed, preserving the original detail and quality.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - A tensor of masks corresponding to the removed backgrounds, highlighting the precision in detail preservation.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RemBgUltra:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "image": ("IMAGE",),
                "detail_range": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01}),
                "process_detail": ("BOOLEAN", {"default": True}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "rembg_ultra"
    CATEGORY = '😺dzNodes/LayerMask'
  
    def rembg_ultra(self, image, detail_range, black_point, white_point, process_detail):
        ret_images = []
        ret_masks = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            i = pil2tensor(tensor2pil(i).convert('RGB'))
            orig_image = tensor2pil(i).convert('RGB')
            _mask = RMBG(orig_image)
            if process_detail:
                _mask = tensor2pil(mask_edge_detail(i, pil2tensor(_mask), detail_range, black_point, white_point))
            ret_image = RGB2RGBA(orig_image, _mask.convert('L'))
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

```
