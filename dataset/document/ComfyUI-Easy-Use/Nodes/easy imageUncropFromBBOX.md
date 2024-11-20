---
tags:
- Crop
- Image
- ImageTransformation
---

# imageUncropFromBBOX
## Documentation
- Class name: `easy imageUncropFromBBOX`
- Category: `EasyUse/Image`
- Output node: `False`

The `imageUncropFromBBOX` node is designed to reverse the cropping process of an image by using bounding box information. It restores the cropped image back to its original dimensions, optionally blending borders and adjusting for square masks, making it useful for tasks that require the original context of the image for further processing or visualization.
## Input types
### Required
- **`original_image`**
    - The original image before any cropping was applied. It serves as the reference for uncropping the cropped image back to its original dimensions.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`crop_image`**
    - The cropped version of the original image that needs to be uncropped. This image will be adjusted based on the bounding box and other parameters to match the original image's dimensions.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`bbox`**
    - The bounding box information used to crop the original image. This data is essential for accurately uncropping the image.
    - Comfy dtype: `BBOX`
    - Python dtype: `Tuple[int, int, int, int]`
- **`border_blending`**
    - A float value that determines the intensity of border blending when uncropping the image. It helps in smoothing the transition between the cropped and uncropped areas.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`use_square_mask`**
    - A boolean flag indicating whether a square mask should be applied during the uncropping process. This affects the shape of the uncropped area.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`optional_mask`**
    - An optional mask that can be applied to the uncropped image for additional processing or visualization purposes.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The uncropped image, restored to its original dimensions with optional border blending and mask adjustments.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class imageUncropFromBBOX:
    @classmethod
    def INPUT_TYPES(s):
        return {
          "required": {
              "original_image": ("IMAGE",),
              "crop_image": ("IMAGE",),
              "bbox": ("BBOX",),
              "border_blending": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},),
              "use_square_mask": ("BOOLEAN", {"default": True}),
          },
          "optional":{
            "optional_mask": ("MASK",)
          }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "uncrop"
    CATEGORY = "EasyUse/Image"

    def bbox_check(self, bbox, target_size=None):
      if not target_size:
        return bbox

      new_bbox = (
        bbox[0],
        bbox[1],
        min(target_size[0] - bbox[0], bbox[2]),
        min(target_size[1] - bbox[1], bbox[3]),
      )
      return new_bbox

    def bbox_to_region(self, bbox, target_size=None):
      bbox = self.bbox_check(bbox, target_size)
      return (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])

    def uncrop(self, original_image, crop_image, bbox, border_blending, use_square_mask, optional_mask=None):
      def inset_border(image, border_width=20, border_color=(0)):
        width, height = image.size
        bordered_image = Image.new(image.mode, (width, height), border_color)
        bordered_image.paste(image, (0, 0))
        draw = ImageDraw.Draw(bordered_image)
        draw.rectangle((0, 0, width - 1, height - 1), outline=border_color, width=border_width)
        return bordered_image

      if len(original_image) != len(crop_image):
        raise ValueError(
          f"The number of original_images ({len(original_image)}) and cropped_images ({len(crop_image)}) should be the same")

        # Ensure there are enough bboxes, but drop the excess if there are more bboxes than images
      if len(bbox) > len(original_image):
        print(f"Warning: Dropping excess bounding boxes. Expected {len(original_image)}, but got {len(bbox)}")
        bbox = bbox[:len(original_image)]
      elif len(bbox) < len(original_image):
        raise ValueError("There should be at least as many bboxes as there are original and cropped images")


      out_images = []

      for i in range(len(original_image)):
        img = tensor2pil(original_image[i])
        crop = tensor2pil(crop_image[i])
        _bbox = bbox[i]

        bb_x, bb_y, bb_width, bb_height = _bbox
        paste_region = self.bbox_to_region((bb_x, bb_y, bb_width, bb_height), img.size)
        
        # rescale the crop image to fit the paste_region
        crop = crop.resize((round(paste_region[2] - paste_region[0]), round(paste_region[3] - paste_region[1])))
        crop_img = crop.convert("RGB")

        # border blending
        if border_blending > 1.0:
          border_blending = 1.0
        elif border_blending < 0.0:
          border_blending = 0.0

        blend_ratio = (max(crop_img.size) / 2) * float(border_blending)
        blend = img.convert("RGBA")

        if use_square_mask:
          mask = Image.new("L", img.size, 0)
          mask_block = Image.new("L", (paste_region[2] - paste_region[0], paste_region[3] - paste_region[1]), 255)
          mask_block = inset_border(mask_block, round(blend_ratio / 2), (0))
          mask.paste(mask_block, paste_region)
        else:
          if optional_mask is None:
            raise ValueError("optional_mask is required when use_square_mask is False")
          original_mask = tensor2pil(optional_mask)
          original_mask = original_mask.resize((paste_region[2] - paste_region[0], paste_region[3] - paste_region[1]))
          mask = Image.new("L", img.size, 0)
          mask.paste(original_mask, paste_region)

        mask = mask.filter(ImageFilter.BoxBlur(radius=blend_ratio / 4))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blend_ratio / 4))

        blend.paste(crop_img, paste_region)
        blend.putalpha(mask)

        img = Image.alpha_composite(img.convert("RGBA"), blend)
        out_images.append(img.convert("RGB"))

      output_images = torch.cat([pil2tensor(img) for img in out_images], dim=0)
      return (output_images,)

```
