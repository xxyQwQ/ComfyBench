---
tags:
- ImageTransformation
- VisualEffects
---

# Image Grab PIL
## Documentation
- Class name: `ImageGrabPIL`
- Category: `KJNodes/experimental`
- Output node: `False`

The ImageGrabPIL node is designed for processing and transforming images into a standardized format suitable for further analysis or processing. It focuses on converting images to a consistent size, color mode, and numerical representation, ensuring they are ready for tasks such as machine learning model input or image manipulation operations.
## Input types
### Required
- **`x`**
    - The 'x' input specifies the starting x-coordinate for the region of the image to be captured. It is essential for defining the area of interest within the larger image context.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`y`**
    - The 'y' input specifies the starting y-coordinate for the region of the image to be captured. It plays a critical role in pinpointing the area of interest on the vertical axis within the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`width`**
    - The 'width' parameter determines the width of the region to be captured from the source image. It is crucial for specifying the size of the area of interest horizontally.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The 'height' parameter determines the height of the region to be captured from the source image. It is vital for defining the vertical size of the area of interest.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`num_frames`**
    - The 'num_frames' parameter indicates the number of frames to capture in a sequence. It is essential for tasks requiring multiple, consecutive captures from the source image, such as creating animations or analyzing changes over time.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`delay`**
    - The 'delay' parameter specifies the time delay between capturing consecutive frames. It is crucial for controlling the timing in sequences where multiple frames are captured, allowing for precise timing adjustments.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The 'image' output is the captured or processed portion of the input image, ready for further processing or analysis.
    - Python dtype: `PIL.Image or similar`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [VAEEncode](../../Comfy/Nodes/VAEEncode.md)
    - [CannyEdgePreprocessor](../../comfyui_controlnet_aux/Nodes/CannyEdgePreprocessor.md)
    - [PixelPerfectResolution](../../comfyui_controlnet_aux/Nodes/PixelPerfectResolution.md)



## Source code
```python
class ImageGrabPIL:

    @classmethod
    def IS_CHANGED(cls):

        return

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "screencap"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = """
Captures an area specified by screen coordinates.  
Can be used for realtime diffusion with autoqueue.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "x": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
                 "y": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
                 "width": ("INT", {"default": 512,"min": 0, "max": 4096, "step": 1}),
                 "height": ("INT", {"default": 512,"min": 0, "max": 4096, "step": 1}),
                 "num_frames": ("INT", {"default": 1,"min": 1, "max": 255, "step": 1}),
                 "delay": ("FLOAT", {"default": 0.1,"min": 0.0, "max": 10.0, "step": 0.01}),
        },
    } 

    def screencap(self, x, y, width, height, num_frames, delay):
        captures = []
        bbox = (x, y, x + width, y + height)
        
        for _ in range(num_frames):
            # Capture screen
            screen_capture = ImageGrab.grab(bbox=bbox)
            screen_capture_torch = torch.tensor(np.array(screen_capture), dtype=torch.float32) / 255.0
            screen_capture_torch = screen_capture_torch.unsqueeze(0)
            captures.append(screen_capture_torch)
            
            # Wait for a short delay if more than one frame is to be captured
            if num_frames > 1:
                time.sleep(delay)
        
        return (torch.cat(captures, dim=0),)

```
