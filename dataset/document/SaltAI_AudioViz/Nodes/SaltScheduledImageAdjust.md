---
tags:
- Scheduling
---

# Batch Image Scheduled Adjustments
## Documentation
- Class name: `SaltScheduledImageAdjust`
- Category: `SALT/AudioViz/Scheduling/Image`
- Output node: `False`

This node is designed for batch processing of images, allowing for scheduled adjustments to various image attributes such as brightness, contrast, saturation, sharpness, gaussian blur, and edge enhancement. It enables the dynamic modification of these attributes over a sequence of images, facilitating complex visual effects and enhancements in an automated manner.
## Input types
### Required
- **`images`**
    - The collection of images to be adjusted. This parameter is central to the node's operation, as it determines the base content that will be modified according to the specified schedules.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[Image]`
### Optional
- **`masks`**
    - Optional masks that can be applied to the images for selective adjustment. This allows for more precise control over which areas of the images are affected by the scheduled adjustments.
    - Comfy dtype: `MASK`
    - Python dtype: `List[Image]`
- **`brightness_schedule`**
    - A schedule dictating the brightness adjustments over time. This enables dynamic changes to the brightness of the images, enhancing visual storytelling or compensating for lighting variations.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`contrast_schedule`**
    - A schedule for adjusting the contrast of the images. This can be used to emphasize texture and depth, or to create a specific mood by altering the visual intensity of the images.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`saturation_schedule`**
    - Defines how the color saturation of the images will change over time. This is useful for creating vivid or muted color effects in a sequence of images.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`sharpness_schedule`**
    - A schedule for modifying the sharpness of the images, allowing for the enhancement or softening of details within the images.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`gaussian_blur_schedule`**
    - Determines the intensity of gaussian blur applied to the images over time, useful for creating depth of field effects or focusing attention on specific elements.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`edge_enhance_schedule`**
    - A schedule for the application of edge enhancement, which can be used to accentuate edges and details within the images.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - unknown
    - Python dtype: `unknown`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltScheduledImageAdjust:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "masks": ("MASK",),
                "brightness_schedule": ("LIST", ),
                "contrast_schedule": ("LIST", ),
                "saturation_schedule": ("LIST", ),
                "sharpness_schedule": ("LIST", ),
                "gaussian_blur_schedule": ("LIST", ),
                "edge_enhance_schedule": ("LIST", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "images_adjust"

    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Image"

    def float_value(self, adj_list, idx):
        if isinstance(adj_list, list) and adj_list:
            return adj_list[idx] if idx < len(adj_list) else adj_list[-1]
        else:
            return 1.0

    def images_adjust(
            self, 
            images, 
            masks=[], 
            brightness_schedule=[1.0], 
            contrast_schedule=[1.0], 
            saturation_schedule=[1.0], 
            sharpness_schedule=[1.0], 
            gaussian_blur_schedule=[0.0], 
            edge_enhance_schedule=[0.0]
        ):
        
        adjusted_images = []
        for idx, img in enumerate(images):
            original_pil_image = tensor2pil(img.unsqueeze(0))
            pil_image = original_pil_image.copy() 
            if isinstance(masks, torch.Tensor):
                pil_mask = mask2pil(masks[idx].unsqueeze(0)) if idx < len(masks) else mask2pil(masks[-1].unsqueeze(0))
                pil_mask = pil_mask.resize(original_pil_image.size).convert('L')
            else:
                pil_mask = None

            brightness = self.float_value(brightness_schedule, idx)
            contrast = self.float_value(contrast_schedule, idx)
            saturation = self.float_value(saturation_schedule, idx)
            sharpness = self.float_value(sharpness_schedule, idx)
            gaussian_blur = self.float_value(gaussian_blur_schedule, idx)
            edge_enhance = self.float_value(edge_enhance_schedule, idx)

            if brightness != 1.0:
                pil_image = ImageEnhance.Brightness(pil_image).enhance(brightness)
            
            if contrast != 1.0:
                pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast)

            if saturation != 1.0:
                pil_image = ImageEnhance.Color(pil_image).enhance(saturation)

            if sharpness != 1.0:
                pil_image = ImageEnhance.Sharpness(pil_image).enhance(sharpness)

            if gaussian_blur > 0.0:
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=gaussian_blur))

            if edge_enhance > 0.0:
                edge_enhanced_img = pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
                blend_mask = Image.new("L", pil_image.size, color=int(round(edge_enhance * 255)))
                pil_image = Image.composite(edge_enhanced_img, pil_image, blend_mask)

            if pil_mask:
                pil_image = Image.composite(pil_image, original_pil_image, pil_mask)

            adjusted_images.append(pil2tensor(pil_image))

        return (torch.cat(adjusted_images, dim=0), )

```
