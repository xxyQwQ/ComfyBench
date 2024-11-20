---
tags:
- Crop
- Image
- ImageTransformation
---

# imageCropFromMask
## Documentation
- Class name: `easy imageCropFromMask`
- Category: `EasyUse/Image`
- Output node: `False`

The `imageCropFromMask` node is designed for cropping images based on a provided mask. It utilizes the mask to determine the relevant area of the image to retain, effectively isolating and extracting the portion of the image that corresponds to the mask's coverage.
## Input types
### Required
- **`image`**
    - The `image` parameter represents the image to be cropped. It is essential for determining the area of interest that will be retained after the cropping process, based on the mask's outline.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask`**
    - The `mask` parameter is crucial for defining the area of the image to be cropped. It acts as a guide for isolating the specific portion of the image that should be retained, based on the mask's coverage.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`image_crop_multi`**
    - This parameter specifies the multiplier to adjust the crop size relative to the mask's bounding box, affecting the final cropped image size.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`mask_crop_multi`**
    - This parameter adjusts the size of the mask used for cropping, influencing the area of the image that gets cropped.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`bbox_smooth_alpha`**
    - This parameter controls the smoothing of bounding box size changes over time, affecting the consistency of the crop dimensions.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`crop_image`**
    - Comfy dtype: `IMAGE`
    - The cropped portion of the original image, as determined by the mask and cropping parameters.
    - Python dtype: `torch.Tensor`
- **`crop_mask`**
    - Comfy dtype: `MASK`
    - The cropped portion of the mask, corresponding to the cropped area of the image.
    - Python dtype: `torch.Tensor`
- **`bbox`**
    - Comfy dtype: `BBOX`
    - The bounding box coordinates of the cropped area within the original image.
    - Python dtype: `List[Tuple[int]]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class imageCropFromMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
          "required": {
              "image": ("IMAGE",),
              "mask": ("MASK",),
              "image_crop_multi": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
              "mask_crop_multi": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
              "bbox_smooth_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
          },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BBOX",)
    RETURN_NAMES = ("crop_image", "crop_mask", "bbox",)
    FUNCTION = "crop"
    CATEGORY = "EasyUse/Image"

    def smooth_bbox_size(self, prev_bbox_size, curr_bbox_size, alpha):
        if alpha == 0:
            return prev_bbox_size
        return round(alpha * curr_bbox_size + (1 - alpha) * prev_bbox_size)

    def smooth_center(self, prev_center, curr_center, alpha=0.5):
        if alpha == 0:
            return prev_center
        return (
            round(alpha * curr_center[0] + (1 - alpha) * prev_center[0]),
            round(alpha * curr_center[1] + (1 - alpha) * prev_center[1])
        )

    def image2mask(self, image):
      return image[:, :, :, 0]

    def mask2image(self, mask):
      return mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)

    def cropimage(self, original_images, masks, crop_size_mult, bbox_smooth_alpha):

      bounding_boxes = []
      cropped_images = []

      self.max_bbox_width = 0
      self.max_bbox_height = 0

      # First, calculate the maximum bounding box size across all masks
      curr_max_bbox_width = 0
      curr_max_bbox_height = 0
      for mask in masks:
        _mask = tensor2pil(mask)
        non_zero_indices = np.nonzero(np.array(_mask))
        min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
        min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
        width = max_x - min_x
        height = max_y - min_y
        curr_max_bbox_width = max(curr_max_bbox_width, width)
        curr_max_bbox_height = max(curr_max_bbox_height, height)

      # Smooth the changes in the bounding box size
      self.max_bbox_width = self.smooth_bbox_size(self.max_bbox_width, curr_max_bbox_width, bbox_smooth_alpha)
      self.max_bbox_height = self.smooth_bbox_size(self.max_bbox_height, curr_max_bbox_height, bbox_smooth_alpha)

      # Apply the crop size multiplier
      self.max_bbox_width = round(self.max_bbox_width * crop_size_mult)
      self.max_bbox_height = round(self.max_bbox_height * crop_size_mult)
      bbox_aspect_ratio = self.max_bbox_width / self.max_bbox_height

      # Then, for each mask and corresponding image...
      for i, (mask, img) in enumerate(zip(masks, original_images)):
        _mask = tensor2pil(mask)
        non_zero_indices = np.nonzero(np.array(_mask))
        min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
        min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])

        # Calculate center of bounding box
        center_x = np.mean(non_zero_indices[1])
        center_y = np.mean(non_zero_indices[0])
        curr_center = (round(center_x), round(center_y))

        # If this is the first frame, initialize prev_center with curr_center
        if not hasattr(self, 'prev_center'):
          self.prev_center = curr_center

        # Smooth the changes in the center coordinates from the second frame onwards
        if i > 0:
          center = self.smooth_center(self.prev_center, curr_center, bbox_smooth_alpha)
        else:
          center = curr_center

        # Update prev_center for the next frame
        self.prev_center = center

        # Create bounding box using max_bbox_width and max_bbox_height
        half_box_width = round(self.max_bbox_width / 2)
        half_box_height = round(self.max_bbox_height / 2)
        min_x = max(0, center[0] - half_box_width)
        max_x = min(img.shape[1], center[0] + half_box_width)
        min_y = max(0, center[1] - half_box_height)
        max_y = min(img.shape[0], center[1] + half_box_height)

        # Append bounding box coordinates
        bounding_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y))

        # Crop the image from the bounding box
        cropped_img = img[min_y:max_y, min_x:max_x, :]

        # Calculate the new dimensions while maintaining the aspect ratio
        new_height = min(cropped_img.shape[0], self.max_bbox_height)
        new_width = round(new_height * bbox_aspect_ratio)

        # Resize the image
        resize_transform = Resize((new_height, new_width))
        resized_img = resize_transform(cropped_img.permute(2, 0, 1))

        # Perform the center crop to the desired size
        crop_transform = CenterCrop((self.max_bbox_height, self.max_bbox_width))  # swap the order here if necessary
        cropped_resized_img = crop_transform(resized_img)

        cropped_images.append(cropped_resized_img.permute(1, 2, 0))

      return cropped_images, bounding_boxes

    def crop(self, image, mask, image_crop_multi, mask_crop_multi, bbox_smooth_alpha):
      cropped_images, bounding_boxes = self.cropimage(image, mask, image_crop_multi, bbox_smooth_alpha)
      cropped_mask_image, _ = self.cropimage(self.mask2image(mask), mask, mask_crop_multi, bbox_smooth_alpha)

      cropped_image_out = torch.stack(cropped_images, dim=0)
      cropped_mask_out = torch.stack(cropped_mask_image, dim=0)

      return (cropped_image_out, cropped_mask_out[:, :, :, 0], bounding_boxes)

```
