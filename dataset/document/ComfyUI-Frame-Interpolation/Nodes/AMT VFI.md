---
tags:
- AnimationScheduling
- FrameInterpolation
- VisualEffects
---

# AMT VFI
## Documentation
- Class name: `AMT VFI`
- Category: `ComfyUI-Frame-Interpolation/VFI`
- Output node: `False`

The AMT_VFI node is designed for advanced motion transfer and frame interpolation in video processing. It leverages deep learning models to analyze and synthesize frames, achieving high-quality video frame interpolation by understanding and replicating the motion between consecutive frames.
## Input types
### Required
- **`ckpt_name`**
    - The checkpoint name for the model, selecting the specific pre-trained model for frame interpolation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`frames`**
    - The sequence of frames to be interpolated, serving as the input for the frame interpolation process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`clear_cache_after_n_frames`**
    - Controls how often the cache is cleared during the frame interpolation process, optimizing memory usage.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`multiplier`**
    - The factor by which the frame rate is increased, determining the number of frames generated between each pair of input frames.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`optional_interpolation_states`**
    - Optional states for interpolation, allowing for customization of the interpolation process.
    - Comfy dtype: `INTERPOLATION_STATES`
    - Python dtype: `InterpolationStateList`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output interpolated frames, showcasing the node's capability in enhancing video fluidity and detail.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class AMT_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (list(CKPT_CONFIGS.keys()), ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 1, "min": 1, "max": 100}),
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
        model_path = load_file_from_direct_url(MODEL_TYPE, f"https://huggingface.co/lalala125/AMT/resolve/main/{ckpt_name}")
        ckpt_config = CKPT_CONFIGS[ckpt_name]

        interpolation_model = ckpt_config["network"](**ckpt_config["params"])
        interpolation_model.load_state_dict(torch.load(model_path)["state_dict"])
        interpolation_model.eval().to(get_torch_device())

        frames = preprocess_frames(frames)
        padder = InputPadder(frames.shape, 16)
        frames = padder.pad(frames)
        
        def return_middle_frame(frame_0, frame_1, timestep, model):
            return model(
                frame_0, 
                frame_1,
                embt=torch.FloatTensor([timestep] * frame_0.shape[0]).view(frame_0.shape[0], 1, 1, 1).to(get_torch_device()),
                scale_factor=1.0,
                eval=True
            )["imgt_pred"]
        
        args = [interpolation_model]
        out = generic_frame_loop(type(self).__name__, frames, clear_cache_after_n_frames, multiplier, return_middle_frame, *args, 
                               interpolation_states=optional_interpolation_states, dtype=torch.float32)
        out = padder.unpad(out)
        out = postprocess_frames(out)
        return (out,)

```
