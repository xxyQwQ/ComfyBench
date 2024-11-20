---
tags:
- DepthMap
- DepthMapEstimation
- Image
- NormalMap
---

# Metric3D Normal Map
## Documentation
- Class name: `Metric3D-NormalMapPreprocessor`
- Category: `ControlNet Preprocessors/Normal and Depth Estimators`
- Output node: `False`

This node preprocesses images for normal map estimation using a 3D metric model. It leverages a pre-trained Metric3DDetector model, configurable with different backbone architectures and intrinsic camera parameters, to compute normal maps from input images. The node aims to enhance depth and normal estimation tasks by providing detailed normal maps, which are crucial for accurate 3D reconstruction and analysis.
## Input types
### Required
- **`image`**
    - The input image to be processed for normal map estimation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
### Optional
- **`backbone`**
    - Specifies the backbone architecture for the Metric3DDetector model. Different backbones offer varying levels of detail and accuracy in the normal map estimation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`fx`**
    - The focal length of the camera in the x-axis, used to calibrate the Metric3DDetector model for accurate normal map estimation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`fy`**
    - The focal length of the camera in the y-axis, essential for calibrating the model to accurately estimate normal maps.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`resolution`**
    - The resolution to which the input image is scaled before processing. This parameter can affect the detail level of the estimated normal map.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is an image representing the estimated normal map, which visualizes the orientation of surfaces in the scene.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class Metric3D_Normal_Map_Preprocessor:
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
        cb = lambda image, **kwargs: model(image, **kwargs)[1]
        out = common_annotator_call(cb, image, resolution=resolution, fx=fx, fy=fy, depth_and_normal=True)
        del model
        return (out, )

```
