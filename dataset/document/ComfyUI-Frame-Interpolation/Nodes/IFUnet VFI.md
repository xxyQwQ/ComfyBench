---
tags:
- AnimationScheduling
- FrameInterpolation
- VisualEffects
---

# IFUnet VFI
## Documentation
- Class name: `IFUnet VFI`
- Category: `ComfyUI-Frame-Interpolation/VFI`
- Output node: `False`

The IFUnet_VFI node specializes in video frame interpolation, leveraging deep learning models to predict and generate intermediate frames between existing frames in a video sequence. This process enhances video fluidity and can be used to increase the frame rate of videos, improve slow-motion effects, or restore missing frames in damaged video files.
## Input types
### Required
- **`ckpt_name`**
    - Specifies the checkpoint name for the model to be used in the frame interpolation process, determining the specific pre-trained model configuration.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`frames`**
    - A sequence of images representing the video frames between which the interpolation will occur, serving as the input for generating intermediate frames.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[Image]`
- **`clear_cache_after_n_frames`**
    - Controls the frequency of cache clearing to manage memory usage during the interpolation process, optimizing performance.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`multiplier`**
    - Defines the factor by which the frame rate is to be increased, indicating the number of intermediate frames to be generated.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`scale_factor`**
    - Determines the scaling factor applied to the frames during the interpolation process, affecting the resolution and size of the output frames.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`ensemble`**
    - A boolean flag that enables or disables the use of ensemble methods for frame interpolation, potentially improving the quality of the output.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`optional_interpolation_states`**
    - Optional states for managing the interpolation process, allowing for advanced control over frame selection and processing.
    - Comfy dtype: `INTERPOLATION_STATES`
    - Python dtype: `InterpolationStates`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image sequence after interpolation, representing the enhanced video with additional frames inserted.
    - Python dtype: `List[Image]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class IFUnet_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (CKPT_NAMES, ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 1000}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100, "step": 0.1}),
                "ensemble": ("BOOLEAN", {"default":True})
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", )
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"

    def vfi(
        self,
        ckpt_name: typing.AnyStr, 
        frames: torch.Tensor, 
        clear_cache_after_n_frames: typing.SupportsInt = 1,
        multiplier: typing.SupportsInt = 2,
        scale_factor: typing.SupportsFloat = 1.0,
        ensemble: bool = True,
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs
    ):
        from .IFUNet_arch import IFUNetModel
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        interpolation_model = IFUNetModel()
        interpolation_model.load_state_dict(torch.load(model_path))
        interpolation_model.eval().to(get_torch_device())
        frames = preprocess_frames(frames)
        
        def return_middle_frame(frame_0, frame_1, timestep, model, scale_factor, ensemble):
            return model(frame_0, frame_1, timestep=timestep, scale=scale_factor, ensemble=ensemble)
        
        args = [interpolation_model, scale_factor, ensemble]
        out = postprocess_frames(
            generic_frame_loop(type(self).__name__, frames, clear_cache_after_n_frames, multiplier, return_middle_frame, *args, 
                               interpolation_states=optional_interpolation_states, dtype=torch.float32)
        )
        return (out,)

```
