---
tags:
- DepthMap
- DepthMapEstimation
- Image
---

# Depth Anything V2 - Relative
## Documentation
- Class name: `DepthAnythingV2Preprocessor`
- Category: `ControlNet Preprocessors/Normal and Depth Estimators`
- Output node: `False`

This node is designed to preprocess images for depth estimation, utilizing a selection of pretrained models to generate depth maps. It allows for the customization of the model checkpoint and resolution, catering to different levels of detail and accuracy in depth estimation.
## Input types
### Required
- **`image`**
    - The input image to be processed for depth estimation. It is the primary data upon which the depth map will be generated.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
### Optional
- **`ckpt_name`**
    - The name of the checkpoint file for the pretrained model to be used in depth estimation. This selection enables the use of different model architectures or training datasets for varied depth estimation needs.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`resolution`**
    - The resolution at which the depth map should be generated. Higher resolutions can provide more detailed depth estimations but may require more computational resources.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output depth map image, providing a visual representation of the estimated depths across the input image.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class Depth_Anything_V2_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            ckpt_name=(["depth_anything_v2_vitg.pth", "depth_anything_v2_vitl.pth", "depth_anything_v2_vitb.pth", "depth_anything_v2_vits.pth"], {"default": "depth_anything_v2_vitl.pth"})
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, ckpt_name, resolution=512, **kwargs):
        from controlnet_aux.depth_anything_v2 import DepthAnythingV2Detector

        model = DepthAnythingV2Detector.from_pretrained(filename=ckpt_name).to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution, max_depth=1)
        del model
        return (out, )

```
