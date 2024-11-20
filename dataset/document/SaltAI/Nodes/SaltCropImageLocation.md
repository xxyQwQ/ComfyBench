---
tags:
- Crop
- Image
- ImageTransformation
---

# Crop Batch Image Location
## Documentation
- Class name: `SaltCropImageLocation`
- Category: `SALT/Image/Process`
- Output node: `False`

This node specializes in identifying and extracting specific image locations for cropping within a batch processing environment. It facilitates precise, location-based cropping operations on multiple images, enhancing the efficiency and accuracy of batch image manipulation tasks.
## Input types
### Required
- **`images`**
    - The collection of images to be processed for location-based cropping. This parameter is crucial for determining the specific areas within each image that need to be cropped, based on the provided location data.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[Image]`
- **`top`**
    - Specifies the top boundary for the cropping operation. It defines the starting vertical point from which the image will be cropped.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`left`**
    - Specifies the left boundary for the cropping operation. It defines the starting horizontal point from which the image will be cropped.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`right`**
    - Specifies the right boundary for the cropping operation, determining the end horizontal point for the crop.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`bottom`**
    - Specifies the bottom boundary for the cropping operation, determining the end vertical point for the crop.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`crop_data_batch`**
    - Optional parameter that includes pre-defined cropping data for batch processing. When provided, it can override manual boundary settings for more automated cropping.
    - Comfy dtype: `CROP_DATA_BATCH`
    - Python dtype: `Optional[List[Tuple[int, int, int, int]]]`
## Output types
- **`images`**
    - Comfy dtype: `IMAGE`
    - The resulting images after the cropping operation has been applied, reflecting the specified location-based adjustments.
    - Python dtype: `List[Image]`
- **`crop_data_batch`**
    - Comfy dtype: `CROP_DATA_BATCH`
    - A batch of crop data corresponding to each processed image, detailing the specific crop locations and dimensions applied.
    - Python dtype: `List[Tuple[int, int, int, int]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltCropImageLocation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "top": ("INT", {"default":0, "max": 10000000, "min":0, "step":1}),
                "left": ("INT", {"default":0, "max": 10000000, "min":0, "step":1}),
                "right": ("INT", {"default":256, "max": 10000000, "min":0, "step":1}),
                "bottom": ("INT", {"default":256, "max": 10000000, "min":0, "step":1}),
            },
            "optional": {
                "crop_data_batch": ("CROP_DATA_BATCH",)
            }
        }
    
    RETURN_TYPES = ("IMAGE", "CROP_DATA_BATCH")
    RETURN_NAMES = ("images", "crop_data_batch")

    FUNCTION = "crop"
    CATEGORY = f"{NAME}/Image/Process"

    def crop(self, images, top=0, left=0, right=256, bottom=256, crop_data_batch=None):
        cropped_images = []
        crop_data_list = []
        master_size = None
        
        for i, image in enumerate(images):
            image = tensor2pil(image)
            img_width, img_height = image.size

            if crop_data_batch is not None:
                if len(crop_data_batch) >= i:
                    crop_size, (crop_left, crop_top, crop_right, crop_bottom) = crop_data_batch[i]
                else:
                    cropped_images.append(pil2tensor(image))
                    crop_data_list.append((image.size, (0, 0, image.size[0], image.size[1])))
                    continue
            else:
                crop_top = max(top, 0)
                crop_left = max(left, 0)
                crop_bottom = min(bottom, img_height)
                crop_right = min(right, img_width)

            crop_width = crop_right - crop_left
            crop_height = crop_bottom - crop_top

            if crop_width <= 0 or crop_height <= 0:
                errmsg = "Invalid crop dimensions."
                logger.error(errmsg)
                raise ValueError(errmsg)
            
            crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
            crop_data = (crop.size, (crop_left, crop_top, crop_right, crop_bottom))
            if not master_size:
                master_size = (((crop.size[0] // 8) * 8), ((crop.size[1] // 8) * 8))
            crop = crop.resize(master_size)
            
            cropped_images.append(pil2tensor(crop))
            crop_data_list.append(crop_data)
        
        cropped_images_batch = torch.cat(cropped_images, dim=0)
        return (cropped_images_batch, crop_data_list)

```
