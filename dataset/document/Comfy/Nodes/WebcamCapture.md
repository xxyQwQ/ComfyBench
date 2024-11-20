---
tags:
- Multimedia
---

# Webcam Capture
## Documentation
- Class name: `WebcamCapture`
- Category: `image`
- Output node: `False`

The WebcamCapture node is designed to capture images from a webcam, allowing for the specification of image dimensions and the option to capture based on a queue. It extends the functionality of loading images by integrating webcam support, making it suitable for real-time image processing applications.
## Input types
### Required
- **`image`**
    - Specifies the source from the webcam for image capture. It is essential for initiating the capture process.
    - Comfy dtype: `WEBCAM`
    - Python dtype: `str`
- **`width`**
    - Determines the width of the captured image. A value of 0 indicates no resizing, allowing for customization of the image dimensions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Sets the height of the captured image. Similar to width, a value of 0 means the image will not be resized, providing flexibility in image size.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`capture_on_queue`**
    - Controls whether the image capture is triggered based on a queue. When enabled, it allows for more controlled and efficient image capturing.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The captured image from the webcam. This output can be directly used for further processing or visualization.
    - Python dtype: `numpy.ndarray`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class WebcamCapture(nodes.LoadImage):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("WEBCAM", {}),
                "width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "capture_on_queue": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_capture"

    CATEGORY = "image"

    def load_capture(s, image, **kwargs):
        return super().load_image(folder_paths.get_annotated_filepath(image))

```
