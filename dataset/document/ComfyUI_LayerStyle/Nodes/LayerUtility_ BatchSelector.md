---
tags:
- Batch
- Image
- ImageBatch
---

# LayerUtility: Batch Selector
## Documentation
- Class name: `LayerUtility: BatchSelector`
- Category: `😺dzNodes/LayerUtility/SystemIO`
- Output node: `False`

The BatchSelector node is designed to selectively process and return subsets of images and masks based on specified indices. It enables dynamic selection and manipulation of image batches, facilitating customized processing flows within image editing or generation pipelines.
## Input types
### Required
- **`select`**
    - Specifies the indices of images and masks to be selected for processing. This parameter is crucial for determining which elements of the input batches are operated on, allowing for targeted manipulation or analysis.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`images`**
    - An optional batch of images to be selectively processed based on the indices provided in 'select'. This parameter enables the node to work with varying sets of images, adapting to the needs of the workflow.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`masks`**
    - An optional batch of masks to be selectively processed alongside images, based on the indices provided in 'select'. This allows for parallel manipulation of images and their corresponding masks, enhancing workflow flexibility.
    - Comfy dtype: `MASK`
    - Python dtype: `List[torch.Tensor]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The processed batch of images, selectively compiled based on the specified indices. This output facilitates further image processing or analysis steps within the pipeline.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The processed batch of masks, selectively compiled alongside images, based on the specified indices. This output supports tasks requiring synchronized manipulation of images and masks.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class BatchSelector:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "select": ("STRING", {"default": "0,"},),
            },
            "optional": {
                "images": ("IMAGE",),  #
                "masks": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = 'batch_selector'
    CATEGORY = '😺dzNodes/LayerUtility/SystemIO'

    def batch_selector(self, select, images=None, masks=None
                  ):
        ret_images = []
        ret_masks = []
        empty_image = pil2tensor(Image.new("RGBA", (64, 64), (0, 0, 0, 0)))
        empty_mask = image2mask(Image.new("L", (64, 64), color="black"))

        indexs = extract_numbers(select)
        for i in indexs:
            if images is not None:
                if i < len(images):
                    ret_images.append(images[i].unsqueeze(0))
                else:
                    ret_images.append(images[-1].unsqueeze(0))
            if masks is not None:
                if i < len(masks):
                    ret_masks.append(masks[i].unsqueeze(0))
                else:
                    ret_masks.append(masks[-1].unsqueeze(0))

        if len(ret_images) == 0:
            ret_images.append(empty_image)
        if len(ret_masks) == 0:
            ret_masks.append(empty_mask)

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

```
