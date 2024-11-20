---
tags:
- AnimationScheduling
- FrameInterpolation
- VisualEffects
---

# Sepconv VFI
## Documentation
- Class name: `Sepconv VFI`
- Category: `ComfyUI-Frame-Interpolation/VFI`
- Output node: `False`

The Sepconv VFI node is designed for video frame interpolation using separable convolutional networks. It enhances video quality by interpolating additional frames between existing ones, leveraging deep learning techniques to predict and generate intermediate frames with high accuracy and visual fidelity. This process is crucial for increasing the frame rate of videos, improving slow-motion effects, and enhancing the overall viewing experience.
## Input types
### Required
- **`ckpt_name`**
    - The checkpoint name for the model, specifying the pre-trained weights to be used for frame interpolation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`frames`**
    - The sequence of frames to be interpolated, provided as a batch of images.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`clear_cache_after_n_frames`**
    - Controls the cache clearing mechanism to manage memory usage, specifying after how many frames the cache should be cleared.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`multiplier`**
    - Defines the frame rate multiplier, indicating how many intermediate frames should be generated between each pair of original frames.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`optional_interpolation_states`**
    - Optional states for controlling the interpolation process, allowing for customization of the frame generation.
    - Comfy dtype: `INTERPOLATION_STATES`
    - Python dtype: `InterpolationStateList`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output interpolated frames, enhancing the original video by increasing its frame rate through the addition of generated intermediate frames.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SepconvVFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (CKPT_NAMES, ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 1000})
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
        clear_cache_after_n_frames = 10,
        multiplier: typing.SupportsInt = 2,
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs
    ):
        from .sepconv_enhanced import Network
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        interpolation_model = Network()
        interpolation_model.load_state_dict(torch.load(model_path))
        interpolation_model.eval().to(get_torch_device())
        frames = preprocess_frames(frames)
        
        def return_middle_frame(frame_0, frame_1, timestep, model):
            return model(frame_0, frame_1)
        
        args = [interpolation_model]
        out = postprocess_frames(
            generic_frame_loop(type(self).__name__, frames, clear_cache_after_n_frames, multiplier, return_middle_frame, *args, 
                               interpolation_states=optional_interpolation_states, use_timestep=False, dtype=torch.float32)
        )
        return (out,)

```
