---
tags:
- Batch
- Image
- ImageBatch
---

# Image Batch Manager (SuperBeasts.AI)
## Documentation
- Class name: `Image Batch Manager (SuperBeasts.AI)`
- Category: `SuperBeastsAI/Image`
- Output node: `False`

The Image Batch Manager node is designed to streamline the process of managing and processing batches of images within a workflow. It focuses on optimizing the handling, transformation, and preparation of large sets of images for further processing or analysis, ensuring efficient batch operations and enhancing overall workflow productivity.
## Input types
### Required
- **`width`**
    - The 'width' parameter specifies the desired width for the output images after processing. It is essential for ensuring that all images in the batch conform to a uniform size, facilitating consistent analysis or further processing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The 'height' parameter defines the desired height for the output images after processing. Similar to 'width', it ensures uniformity in the size of images, which is critical for batch processing and subsequent analyses.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`max_images`**
    - The 'max_images' parameter limits the number of images to be processed in the batch. This is crucial for controlling the batch size and ensuring efficient processing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`random_order`**
    - The 'random_order' parameter determines whether the images should be reordered randomly. This can be used to introduce variability in the processing sequence.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`new_manual_order`**
    - The 'new_manual_order' parameter allows for the specification of a new order for the processed images. It enables custom sequencing of the input images, affecting the final arrangement and composition of the output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - This output represents the processed batch of images, ready for further analysis or transformation. It encapsulates the results of the batch management operations, providing a streamlined dataset for subsequent processing stages.
    - Python dtype: `torch.Tensor`
- **`string`**
    - Comfy dtype: `STRING`
    - This output is a string representing the new order of the processed images, providing insight into the arrangement of the batch post-processing.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageBatchManagement:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "order": 1}),
                "height": ("INT", {"default": 768}),
                "max_images": ("INT", {"default": 10}),  # New INT input for maximum number of images
                "random_order": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "new_manual_order": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "reorder"
    CATEGORY = "SuperBeastsAI/Image"

    def reorder(self, width, height, random_order, max_images, **kwargs):
        images = [kwargs["image1"]]  # Start with the required image1 input

        i = 2
        while f"image{i}" in kwargs:
            images.append(kwargs[f"image{i}"])
            i += 1

        if max_images is not None:
            images = images[:max_images]

        # Default order_output if new_manual_order isn't provided or is empty
        order_output = ",".join(str(idx + 1) for idx in range(len(images)))

        # Retrieve and apply new_manual_order if it exists
        if 'new_manual_order' in kwargs and kwargs['new_manual_order']:
            order_indices = [int(idx) - 1 for idx in kwargs['new_manual_order'].split(',') if idx.strip()]
            images = [images[idx] for idx in order_indices if idx < len(images)]
            order_output = kwargs['new_manual_order']

        processed_images = []
        for img in images:
            pil_img = tensor2pil(img)
            resized_cropped_img = resize_and_crop(pil_img, width, height)
            img_tensor = pil2tensor(resized_cropped_img)
            processed_images.append(img_tensor)

        result = torch.cat(processed_images, dim=0) if processed_images else torch.empty(0, 3, height, width)
        return (result, order_output)

```
