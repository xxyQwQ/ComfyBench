---
tags:
- Animation
---

# LivePortraitProcess
## Documentation
- Class name: `LivePortraitProcess`
- Category: `LivePortrait`
- Output node: `False`

The LivePortraitProcess node is designed to transform static portrait images into animated versions by applying a series of image processing and deep learning techniques. It leverages facial landmarks, cropping, resizing, and feature extraction to prepare the source image. Subsequently, it utilizes keypoints information, rotation matrices, and 3D features to animate the portrait based on driving images, adjusting for eye and lip movements, and optionally stitching the results for a seamless animation.
## Input types
### Required
- **`pipeline`**
    - The processing pipeline that encapsulates the entire animation process, including cropping, feature extraction, and animation synthesis.
    - Comfy dtype: `LIVEPORTRAITPIPE`
    - Python dtype: `LivePortraitPipeline`
- **`source_image`**
    - The source image to be animated. It serves as the base for the animation process, where facial features are detected and used to guide the transformation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `numpy.ndarray`
- **`driving_images`**
    - A sequence of images that dictate the animation's movements. These images provide the dynamic expressions and orientations to be transferred to the source portrait.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[numpy.ndarray]`
- **`dsize`**
    - The desired size for the output images, affecting how the source and driving images are cropped and resized.
    - Comfy dtype: `INT`
    - Python dtype: `Tuple[int, int]`
- **`scale`**
    - A scaling factor applied during the cropping process to adjust the emphasis on the central features of the portrait.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`vx_ratio`**
    - The horizontal ratio used to fine-tune the cropping area, focusing on the central facial features.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`vy_ratio`**
    - The vertical ratio used to fine-tune the cropping area, focusing on the central facial features.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`lip_zero`**
    - A flag indicating whether to neutralize lip movements in the animation, keeping the lips static.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`eye_retargeting`**
    - A flag to enable or disable eye movement retargeting in the animation, allowing for dynamic eye expressions.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`eyes_retargeting_multiplier`**
    - A multiplier applied to the eye movement retargeting, allowing for exaggerated or subdued eye animations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`lip_retargeting`**
    - A flag to enable or disable lip movement retargeting in the animation, allowing for dynamic lip expressions.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`lip_retargeting_multiplier`**
    - A multiplier applied to the lip movement retargeting, allowing for exaggerated or subdued lip animations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`stitching`**
    - A flag indicating whether to apply stitching to the animated frames, ensuring seamless transitions between expressions.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`relative`**
    - A flag that determines whether the animation should be based on relative movements from the driving images or absolute positions.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`onnx_device`**
    - Specifies the computation device (e.g., CUDA for GPU) for running the ONNX models involved in the animation process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`cropped_images`**
    - Comfy dtype: `IMAGE`
    - A list of cropped and processed images ready for animation, representing intermediate steps in the animation pipeline.
    - Python dtype: `List[numpy.ndarray]`
- **`full_images`**
    - Comfy dtype: `IMAGE`
    - The final list of animated portraits, fully processed and ready for display or further use.
    - Python dtype: `List[numpy.ndarray]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LivePortraitProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {

            "pipeline": ("LIVEPORTRAITPIPE",),
            "source_image": ("IMAGE",),
            "driving_images": ("IMAGE",),
            "dsize": ("INT", {"default": 512, "min": 64, "max": 2048}),
            "scale": ("FLOAT", {"default": 2.3, "min": 1.0, "max": 4.0, "step": 0.01}),
            "vx_ratio": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            "vy_ratio": ("FLOAT", {"default": -0.125, "min": -1.0, "max": 1.0, "step": 0.01}),
            "lip_zero": ("BOOLEAN", {"default": True}),
            "eye_retargeting": ("BOOLEAN", {"default": False}),
            "eyes_retargeting_multiplier": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.001}),
            "lip_retargeting": ("BOOLEAN", {"default": False}),
            "lip_retargeting_multiplier": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.001}),
            "stitching": ("BOOLEAN", {"default": True}),
            "relative": ("BOOLEAN", {"default": True}),
            },
            "optional": {
               "onnx_device": (
                    [
                        'CPU',
                        'CUDA',                        
                    ], {
                        "default": 'CPU'
                    }),
            }


        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("cropped_images", "full_images",)
    FUNCTION = "process"
    CATEGORY = "LivePortrait"

    def process(self, source_image, driving_images, dsize, scale, vx_ratio, vy_ratio, pipeline, 
                lip_zero, eye_retargeting, lip_retargeting, stitching, relative, eyes_retargeting_multiplier, lip_retargeting_multiplier, onnx_device='CUDA'):
        source_image_np = (source_image * 255).byte().numpy()
        driving_images_np = (driving_images * 255).byte().numpy()

        crop_cfg = CropConfig(
            dsize = dsize,
            scale = scale,
            vx_ratio = vx_ratio,
            vy_ratio = vy_ratio,
            )
        
        cropper = Cropper(crop_cfg=crop_cfg, provider=onnx_device)
        pipeline.cropper = cropper
        pipeline.live_portrait_wrapper.cfg.flag_eye_retargeting = eye_retargeting
        pipeline.live_portrait_wrapper.cfg.eyes_retargeting_multiplier = eyes_retargeting_multiplier
        pipeline.live_portrait_wrapper.cfg.flag_lip_retargeting = lip_retargeting
        pipeline.live_portrait_wrapper.cfg.lip_retargeting_multiplier = lip_retargeting_multiplier
        pipeline.live_portrait_wrapper.cfg.flag_stitching = stitching
        pipeline.live_portrait_wrapper.cfg.flag_relative = relative
        pipeline.live_portrait_wrapper.cfg.flag_lip_zero = lip_zero
      
        cropped_out_list = []
        full_out_list = []
        for img in source_image_np:
            cropped_frames, full_frame = pipeline.execute(img, driving_images_np)
            cropped_tensors = [torch.from_numpy(np_array) for np_array in cropped_frames]
            cropped_tensors_out = torch.stack(cropped_tensors) / 255
            cropped_tensors_out = cropped_tensors_out.cpu().float()

            full_tensors = [torch.from_numpy(np_array) for np_array in full_frame]
            full_tensors_out = torch.stack(full_tensors) / 255
            full_tensors_out = full_tensors_out.cpu().float()

            cropped_out_list.append(cropped_tensors_out)
            full_out_list.append(full_tensors_out)

        cropped_tensors_out = torch.cat(cropped_out_list, dim=0)
        full_tensors_out = torch.cat(full_out_list, dim=0)

        return (cropped_tensors_out, full_tensors_out)

```
