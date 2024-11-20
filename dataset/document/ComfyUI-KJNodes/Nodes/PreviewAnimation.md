---
tags:
- AnimationScheduling
---

# Preview Animation
## Documentation
- Class name: `PreviewAnimation`
- Category: `KJNodes/image`
- Output node: `True`

The PreviewAnimation node is designed to generate a preview animation from a sequence of images, optionally applying masks to each frame. It allows for the customization of the animation's frames per second (fps), enabling the creation of smooth transitions and dynamic visualizations tailored to specific requirements.
## Input types
### Required
- **`fps`**
    - Specifies the frames per second for the animation, controlling the speed at which the images are displayed.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
### Optional
- **`images`**
    - A list of images to be included in the animation. Each image in the sequence contributes to the overall animation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[Image]`
- **`masks`**
    - Optional masks to apply to each image in the sequence, allowing for selective visibility or effects on the animation frames.
    - Comfy dtype: `MASK`
    - Python dtype: `List[Image]`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class PreviewAnimation:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    methods = {"default": 4, "fastest": 0, "slowest": 6}
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                     "fps": ("FLOAT", {"default": 8.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                     },
                "optional": {
                    "images": ("IMAGE", ),
                    "masks": ("MASK", ),
                },
            }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "KJNodes/image"

    def preview(self, fps, images=None, masks=None):
        filename_prefix = "AnimPreview"
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        results = list()

        pil_images = []

        if images is not None and masks is not None:
            for image in images:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                pil_images.append(img)
            for mask in masks:
                if pil_images: 
                    mask_np = mask.cpu().numpy()
                    mask_np = np.clip(mask_np * 255, 0, 255).astype(np.uint8)  # Convert to values between 0 and 255
                    mask_img = Image.fromarray(mask_np, mode='L')
                    img = pil_images.pop(0)  # Remove and get the first image
                    img = img.convert("RGBA")  # Convert base image to RGBA

                    # Create a new RGBA image based on the grayscale mask
                    rgba_mask_img = Image.new("RGBA", img.size, (255, 255, 255, 255))
                    rgba_mask_img.putalpha(mask_img)  # Use the mask image as the alpha channel

                    # Composite the RGBA mask onto the base image
                    composited_img = Image.alpha_composite(img, rgba_mask_img)
                    pil_images.append(composited_img)  # Add the composited image back

        elif images is not None and masks is None:
            for image in images:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                pil_images.append(img)

        elif masks is not None and images is None:
            for mask in masks:
                mask_np = 255. * mask.cpu().numpy()
                mask_img = Image.fromarray(np.clip(mask_np, 0, 255).astype(np.uint8))
                pil_images.append(mask_img)
        else:
            print("PreviewAnimation: No images or masks provided")
            return { "ui": { "images": results, "animated": (None,), "text": "empty" }}

        num_frames = len(pil_images)

        c = len(pil_images)
        for i in range(0, c, num_frames):
            file = f"{filename}_{counter:05}_.webp"
            pil_images[i].save(os.path.join(full_output_folder, file), save_all=True, duration=int(1000.0/fps), append_images=pil_images[i + 1:i + num_frames], lossless=False, quality=80, method=4)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        animated = num_frames != 1
        return { "ui": { "images": results, "animated": (animated,), "text": [f"{num_frames}x{pil_images[0].size[0]}x{pil_images[0].size[1]}"] } }

```
