---
tags:
- Crop
- Image
- ImageTransformation
---

# Mask Batch Crop Region
## Documentation
- Class name: `SaltMaskCropRegion`
- Category: `SALT/Masking/Process`
- Output node: `False`

This node focuses on cropping regions from a batch of masks based on a specified region type and padding. It dynamically adjusts the crop to focus on areas of interest within each mask, resizing the cropped regions to a uniform size for batch processing. The node is designed to enhance or isolate specific features within masks, facilitating further analysis or processing.
## Input types
### Required
- **`masks`**
    - A batch of masks to be processed for cropping. This parameter is crucial for determining the areas within each mask that will be cropped based on the specified region type and padding.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`padding`**
    - The padding value adds a specified number of pixels around the crop region. This parameter allows for flexibility in the size of the cropped area, ensuring that important features are not clipped.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`region_type`**
    - Specifies the type of region to focus on for cropping, such as 'dominant'. This parameter influences how the cropping algorithm identifies and isolates areas of interest within each mask.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`cropped_masks`**
    - Comfy dtype: `MASK`
    - The batch of masks after cropping, resized to a uniform size for consistency across the batch.
    - Python dtype: `torch.Tensor`
- **`crop_data_batch`**
    - Comfy dtype: `CROP_DATA_BATCH`
    - A list of crop data for each mask, detailing the specific crop dimensions and locations.
    - Python dtype: `list`
- **`top_int`**
    - Comfy dtype: `INT`
    - The top coordinate of the crop region for the first mask in the batch.
    - Python dtype: `int`
- **`left_int`**
    - Comfy dtype: `INT`
    - The left coordinate of the crop region for the first mask in the batch.
    - Python dtype: `int`
- **`right_int`**
    - Comfy dtype: `INT`
    - The right coordinate of the crop region for the first mask in the batch.
    - Python dtype: `int`
- **`bottom_int`**
    - Comfy dtype: `INT`
    - The bottom coordinate of the crop region for the first mask in the batch.
    - Python dtype: `int`
- **`width_int`**
    - Comfy dtype: `INT`
    - The width of the cropped region for the first mask in the batch, providing insight into the size of the cropped area.
    - Python dtype: `int`
- **`height_int`**
    - Comfy dtype: `INT`
    - The height of the cropped region for the first mask in the batch, offering a measure of the cropped area's size.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SaltMaskCropRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "padding": ("INT",{"default": 24, "min": 0, "max": 4096, "step": 1}),
                "region_type": (["dominant", "minority"],),
            }
        }
    
    RETURN_TYPES = ("MASK", "CROP_DATA_BATCH", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cropped_masks", "crop_data_batch", "top_int", "left_int", "right_int", "bottom_int", "width_int", "height_int")
    
    FUNCTION = "mask_crop_region"
    CATEGORY = f"{NAME}/Masking/Process"
    
    def mask_crop_region(self, masks, padding=24, region_type="dominant"):
        N = len(masks)
        cropped_masks = []
        crop_data_list = []
        master_size = None
        
        for n in range(N):
            mask = masks[n]
            mask_pil = tensor2pil(mask)
            if not master_size:
                master_size = mask_pil.size
            region_mask, crop_data = MaskFilters.crop_region(mask_pil, region_type, padding)
            region_mask = region_mask.resize(master_size)
            region_tensor = pil2mask(region_mask)
            cropped_masks.append(region_tensor)
            crop_data_list.append(crop_data)

        cropped_masks_batch = torch.cat(cropped_masks, dim=0)

        # Extract crop data
        top_int, left_int, right_int, bottom_int = crop_data_list[0][1]
        width_int, height_int = cropped_masks_batch.shape[2], cropped_masks_batch.shape[1]

        return (cropped_masks_batch, crop_data_list, top_int, left_int, right_int, bottom_int, width_int, height_int)

```
