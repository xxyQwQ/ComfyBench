---
tags:
- AnimationScheduling
- FrameInterpolation
- VisualEffects
---

# M2M VFI
## Documentation
- Class name: `M2M VFI`
- Category: `ComfyUI-Frame-Interpolation/VFI`
- Output node: `False`

The M2M VFI node is designed for video frame interpolation, utilizing deep learning models to predict intermediate frames between two consecutive frames in a video sequence. This process enhances video fluidity and can be used to increase the frame rate of videos.
## Input types
### Required
- **`ckpt_name`**
    - A checkpoint name specifying the model to be used for frame interpolation. It determines the specific pre-trained model weights to load, directly influencing the interpolation quality and performance.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`frames`**
    - A tensor containing the sequence of frames for which intermediate frames are to be generated. It is the primary input from which the model predicts the missing frames, affecting the overall video smoothness.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`clear_cache_after_n_frames`**
    - Indicates after how many frames the cache should be cleared to prevent memory overflow. This parameter helps manage GPU memory usage efficiently during the interpolation process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`multiplier`**
    - Defines the number of intermediate frames to be generated between each pair of original frames, directly affecting the output video's frame rate.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`optional_interpolation_states`**
    - An optional parameter that allows for the control of frame skipping and other interpolation behaviors, providing flexibility in the frame generation process.
    - Comfy dtype: `INTERPOLATION_STATES`
    - Python dtype: `InterpolationStateList`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output tensor containing the interpolated frames, which increases the frame rate and enhances the smoothness of the input video sequence.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class M2M_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (CKPT_NAMES, ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 1000}),
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
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs
    ):
        from .M2M_arch import M2M_PWC
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        interpolation_model = M2M_PWC()
        interpolation_model.load_state_dict(torch.load(model_path))
        interpolation_model.eval().to(get_torch_device())
        frames = preprocess_frames(frames)
        
        def return_middle_frame(frame_0, frame_1, int_timestep, model):
            tenSteps = [
                torch.FloatTensor([int_timestep] * len(frame_0)).view(len(frame_0), 1, 1, 1).to(get_torch_device())
            ]
            return model(frame_0, frame_1, tenSteps)[0]
        
        args = [interpolation_model]
        out = postprocess_frames(
            generic_frame_loop(type(self).__name__, frames, clear_cache_after_n_frames, multiplier, return_middle_frame, *args, 
                               interpolation_states=optional_interpolation_states, dtype=torch.float32)
        )
        return (out,)

```
