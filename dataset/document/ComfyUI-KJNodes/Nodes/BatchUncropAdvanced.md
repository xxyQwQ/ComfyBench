---
tags:
- Crop
- Image
- ImageTransformation
---

# Batch Uncrop Advanced
## Documentation
- Class name: `BatchUncropAdvanced`
- Category: `KJNodes/masking`
- Output node: `False`

The `BatchUncropAdvanced` node is designed for the advanced uncropping of images in a batch processing context. It integrates functionalities such as handling multiple images, applying border blending, rescaling crops, and utilizing combined or square masks to refine the uncropping process. This node aims to restore the original dimensions of cropped images while maintaining high fidelity to the original content, especially in scenarios where precise alignment and blending are crucial.
## Input types
### Required
- **`original_images`**
    - Specifies the original images before cropping. This input is crucial for the uncrop operation as it provides the reference for restoring the images to their original state.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`cropped_images`**
    - Contains the images that have been cropped. These images are used alongside the original images to perform the uncropping process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`cropped_masks`**
    - Represents the masks of the cropped areas. These masks are essential for identifying the regions within the original images that need to be restored or blended during the uncropping.
    - Comfy dtype: `MASK`
    - Python dtype: `List[torch.Tensor]`
- **`combined_crop_mask`**
    - A single mask that combines all cropped areas. This mask is used to efficiently handle multiple cropped regions in a batch processing scenario.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`bboxes`**
    - The bounding boxes that define the cropped areas within the original images. These are used to accurately place the cropped images back into their original context.
    - Comfy dtype: `BBOX`
    - Python dtype: `List[Tuple[int, int, int, int]]`
- **`border_blending`**
    - Controls the blending of the borders during the uncropping process to ensure a seamless transition between the cropped and original image areas.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`crop_rescale`**
    - Determines whether the cropped images should be rescaled to their original size during the uncropping process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`use_combined_mask`**
    - Indicates whether the combined crop mask should be used for the uncropping process, allowing for more complex uncropping scenarios.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`use_square_mask`**
    - Specifies whether a square mask should be used for uncropping, which can be beneficial in certain contexts for maintaining uniformity.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`combined_bounding_box`**
    - An optional parameter that defines a single bounding box encompassing all cropped areas, used for more advanced uncropping scenarios.
    - Comfy dtype: `BBOX`
    - Python dtype: `Optional[Tuple[int, int, int, int]]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output images after the uncropping process, showcasing the result of the operation with restored or blended areas.
    - Python dtype: `List[torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class BatchUncropAdvanced:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "cropped_images": ("IMAGE",), 
                "cropped_masks": ("MASK",),
                "combined_crop_mask": ("MASK",),
                "bboxes": ("BBOX",),
                "border_blending": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}, ),
                "crop_rescale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "use_combined_mask": ("BOOLEAN", {"default": False}),
                "use_square_mask": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "combined_bounding_box": ("BBOX", {"default": None}),  
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "uncrop"
    CATEGORY = "KJNodes/masking"


    def uncrop(self, original_images, cropped_images, cropped_masks, combined_crop_mask, bboxes, border_blending, crop_rescale, use_combined_mask, use_square_mask, combined_bounding_box = None):
        
        def inset_border(image, border_width=20, border_color=(0)):
            width, height = image.size
            bordered_image = Image.new(image.mode, (width, height), border_color)
            bordered_image.paste(image, (0, 0))
            draw = ImageDraw.Draw(bordered_image)
            draw.rectangle((0, 0, width - 1, height - 1), outline=border_color, width=border_width)
            return bordered_image

        if len(original_images) != len(cropped_images):
            raise ValueError(f"The number of original_images ({len(original_images)}) and cropped_images ({len(cropped_images)}) should be the same")

        # Ensure there are enough bboxes, but drop the excess if there are more bboxes than images
        if len(bboxes) > len(original_images):
            print(f"Warning: Dropping excess bounding boxes. Expected {len(original_images)}, but got {len(bboxes)}")
            bboxes = bboxes[:len(original_images)]
        elif len(bboxes) < len(original_images):
            raise ValueError("There should be at least as many bboxes as there are original and cropped images")

        crop_imgs = tensor2pil(cropped_images)
        input_images = tensor2pil(original_images)
        out_images = []

        for i in range(len(input_images)):
            img = input_images[i]
            crop = crop_imgs[i]
            bbox = bboxes[i]
            
            if use_combined_mask:
                bb_x, bb_y, bb_width, bb_height = combined_bounding_box[0]
                paste_region = bbox_to_region((bb_x, bb_y, bb_width, bb_height), img.size)
                mask = combined_crop_mask[i]
            else:
                bb_x, bb_y, bb_width, bb_height = bbox
                paste_region = bbox_to_region((bb_x, bb_y, bb_width, bb_height), img.size)
                mask = cropped_masks[i]
            
            # scale paste_region
            scale_x = scale_y = crop_rescale
            paste_region = (round(paste_region[0]*scale_x), round(paste_region[1]*scale_y), round(paste_region[2]*scale_x), round(paste_region[3]*scale_y))

            # rescale the crop image to fit the paste_region
            crop = crop.resize((round(paste_region[2]-paste_region[0]), round(paste_region[3]-paste_region[1])))
            crop_img = crop.convert("RGB")

            #border blending
            if border_blending > 1.0:
                border_blending = 1.0
            elif border_blending < 0.0:
                border_blending = 0.0

            blend_ratio = (max(crop_img.size) / 2) * float(border_blending)
            blend = img.convert("RGBA")

            if use_square_mask:
                mask = Image.new("L", img.size, 0)
                mask_block = Image.new("L", (paste_region[2]-paste_region[0], paste_region[3]-paste_region[1]), 255)
                mask_block = inset_border(mask_block, round(blend_ratio / 2), (0))
                mask.paste(mask_block, paste_region)
            else:
                original_mask = tensor2pil(mask)[0]
                original_mask = original_mask.resize((paste_region[2]-paste_region[0], paste_region[3]-paste_region[1]))
                mask = Image.new("L", img.size, 0)
                mask.paste(original_mask, paste_region)

            mask = mask.filter(ImageFilter.BoxBlur(radius=blend_ratio / 4))
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blend_ratio / 4))

            blend.paste(crop_img, paste_region) 
            blend.putalpha(mask)
            
            img = Image.alpha_composite(img.convert("RGBA"), blend)
            out_images.append(img.convert("RGB"))

        return (pil2tensor(out_images),)

```
