---
tags:
- Crop
- Image
- ImageTransformation
---

# Extract Crop Data from Batch
## Documentation
- Class name: `SaltBatchCropDataExtractor`
- Category: `SALT/Masking/Process`
- Output node: `False`

The SaltBatchCropDataExtractor node is designed for extracting specific crop data from a batch based on an index. It processes a collection of crop data, identifying and extracting detailed information about a particular crop region, including its dimensions and position within the original image.
## Input types
### Required
- **`crop_data_batch`**
    - A batch of crop data from which specific crop information is to be extracted. It plays a crucial role in determining the output by specifying which crop region's details are to be retrieved.
    - Comfy dtype: `CROP_DATA_BATCH`
    - Python dtype: `List[Tuple[Tuple[int, int], Tuple[int, int, int, int]]]`
- **`index`**
    - The index within the crop data batch that specifies the particular crop region to extract information from. It is essential for pinpointing the exact crop data to be processed.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`crop_data_batch`**
    - Comfy dtype: `CROP_DATA_BATCH`
    - The extracted crop data for the specified index, including the crop's size and coordinates.
    - Python dtype: `List[Tuple[Tuple[int, int], Tuple[int, int, int, int]]]`
- **`width`**
    - Comfy dtype: `INT`
    - The width of the extracted crop region.
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - The height of the extracted crop region.
    - Python dtype: `int`
- **`top`**
    - Comfy dtype: `INT`
    - The top coordinate of the extracted crop region.
    - Python dtype: `int`
- **`left`**
    - Comfy dtype: `INT`
    - The left coordinate of the extracted crop region.
    - Python dtype: `int`
- **`right`**
    - Comfy dtype: `INT`
    - The right coordinate of the extracted crop region.
    - Python dtype: `int`
- **`bottom`**
    - Comfy dtype: `INT`
    - The bottom coordinate of the extracted crop region.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltBatchCropDataExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crop_data_batch": ("CROP_DATA_BATCH",),
                "index": ("INT", {"min": 0}),
            }
        }

    RETURN_TYPES = ("CROP_DATA_BATCH", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("crop_data_batch", "width", "height", "top", "left", "right", "bottom")

    FUNCTION = "extract"
    CATEGORY = f"{NAME}/Masking/Process"

    def extract(self, crop_data_batch, index):
        if index < 0 or index >= len(crop_data_batch):
            errmsg = "Index out of range"
            logger.error(errmsg)
            raise ValueError(errmsg)

        try:
            crop_size, (crop_left, crop_top, crop_right, crop_bottom) = crop_data_batch[index]
            width = crop_right - crop_left
            height = crop_bottom - crop_top
            single_crop_data = [(crop_size, (crop_left, crop_top, crop_right, crop_bottom))]
            return single_crop_data, width, height, crop_top, crop_left, crop_right, crop_bottom
        except Exception as e:
            logger.warning(e)
            return [((0, 0), (0, 0, 0, 0))], 0, 0, 0, 0, 0, 0

```
