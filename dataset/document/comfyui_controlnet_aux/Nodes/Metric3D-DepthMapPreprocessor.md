---
tags:
- DepthMap
- DepthMapEstimation
- Image
---

# Metric3D Depth Map
## Documentation
- Class name: `Metric3D-DepthMapPreprocessor`
- Category: `ControlNet Preprocessors/Normal and Depth Estimators`
- Output node: `False`

This node preprocesses images for depth map estimation using a 3D metric model. It leverages a configurable backbone architecture and camera intrinsic parameters to enhance the depth estimation process, aiming to provide a detailed depth map for each input image.
## Input types
### Required
- **`image`**
    - The input image to be processed for depth map estimation. The quality and characteristics of the image can significantly impact the accuracy and detail of the resulting depth map, making it a crucial factor in the node's execution.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
### Optional
- **`backbone`**
    - Specifies the backbone model architecture used for depth estimation. The choice of backbone can significantly influence the accuracy and performance of the depth map generation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`fx`**
    - Represents the focal length of the camera along the x-axis. It is a critical parameter for accurately mapping 2D images to 3D space, affecting the scale and perspective of the depth map.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`fy`**
    - Represents the focal length of the camera along the y-axis, essential for correct depth perception and 3D reconstruction from 2D images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`resolution`**
    - The desired resolution for the output depth map, affecting the level of detail and size of the output image. Higher resolutions can provide more detailed depth maps but may require more computational resources.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a depth map image, providing a pixel-wise depth estimation for the input image.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class Metric3D_Depth_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            backbone=(["vit-small", "vit-large", "vit-giant2"], {"default": "vit-small"}),
            fx=("INT", {"default": 1000, 'min': 1, 'max': MAX_RESOLUTION}),
            fy=("INT", {"default": 1000, 'min': 1, 'max': MAX_RESOLUTION})
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, backbone, fx, fy, resolution=512):
        from controlnet_aux.metric3d import Metric3DDetector
        model = Metric3DDetector.from_pretrained(filename=f"metric_depth_{backbone.replace('-', '_')}_800k.pth").to(model_management.get_torch_device())
        cb = lambda image, **kwargs: model(image, **kwargs)[0]
        out = common_annotator_call(cb, image, resolution=resolution, fx=fx, fy=fy, depth_and_normal=True)
        del model
        return (out, )

```
