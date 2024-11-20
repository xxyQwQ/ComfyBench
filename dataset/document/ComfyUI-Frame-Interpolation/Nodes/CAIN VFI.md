---
tags:
- AnimationScheduling
- FrameInterpolation
- VisualEffects
---

# CAIN VFI
## Documentation
- Class name: `CAIN VFI`
- Category: `ComfyUI-Frame-Interpolation/VFI`
- Output node: `False`

The CAIN VFI node is designed for video frame interpolation, leveraging deep learning models to predict intermediate frames between two consecutive frames in a video sequence. It utilizes the CAIN architecture, which focuses on capturing temporal and spatial details through an encoder-decoder structure, to enhance the smoothness and quality of video playback.
## Input types
### Required
- **`ckpt_name`**
    - The checkpoint name specifies the pre-trained model to be used for frame interpolation, enabling the node to load the appropriate weights for the CAIN model.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`frames`**
    - A tensor containing the consecutive frames of a video. These frames are the input to the CAIN model for generating intermediate frames.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`clear_cache_after_n_frames`**
    - This parameter controls the cache clearing mechanism to manage memory usage during the frame interpolation process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`multiplier`**
    - Defines the number of intermediate frames to be generated between each pair of consecutive frames, dictating the smoothness of the output video.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`optional_interpolation_states`**
    - Optional states that can be passed to influence the interpolation process, offering flexibility for different video processing scenarios and potentially affecting the selection of frames for interpolation.
    - Comfy dtype: `INTERPOLATION_STATES`
    - Python dtype: `InterpolationStateList`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a tensor of interpolated frames, enhancing the video's fluidity by filling in the gaps between original frames.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CAIN_VFI:
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
        clear_cache_after_n_frames: typing.SupportsInt = 1,
        multiplier: typing.SupportsInt = 2,
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs
    ):
        from .cain_arch import CAIN
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        sd = torch.load(model_path)["state_dict"]
        sd = {key.replace('module.', ''): value for key, value in sd.items()}


        global interpolation_model
        interpolation_model = CAIN(depth=3)
        interpolation_model.load_state_dict(sd)
        interpolation_model.eval().to(get_torch_device())
        del sd

        frames = preprocess_frames(frames)
    
        
        def return_middle_frame(frame_0, frame_1, timestep, model):
            #CAIN does some direct modifications to input frame tensors so we need to clone them
            return model(frame_0.detach().clone(), frame_1.detach().clone())[0]
        
        args = [interpolation_model]
        out = postprocess_frames(
            generic_frame_loop(type(self).__name__, frames, clear_cache_after_n_frames, multiplier, return_middle_frame, *args, 
                               interpolation_states=optional_interpolation_states, use_timestep=False, dtype=torch.float32)
        )
        return (out,)

```
