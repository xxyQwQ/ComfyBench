---
tags:
- AnimationScheduling
- FrameInterpolation
- VisualEffects
---

# Remove and Interpolate Frames 🎞️🅢🅜
## Documentation
- Class name: `RemoveAndInterpolateFrames`
- Category: `Steerable-Motion`
- Output node: `False`

The RemoveAndInterpolateFrames node is designed to selectively remove specified frames from a sequence of images and interpolate new frames to fill the gaps, ensuring a smooth transition between frames. This process leverages frame interpolation techniques to enhance the fluidity of motion within the sequence, making it particularly useful for applications requiring high-quality video frame manipulation.
## Input types
### Required
- **`images`**
    - The tensor containing a sequence of images from which specific frames will be removed and between which new frames will be interpolated. It plays a crucial role in determining the input and output quality of the interpolation process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`frames_to_drop`**
    - A string representation of a list indicating the indices of frames to be removed from the sequence. This parameter directly influences which frames are interpolated and replaced, impacting the final video output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - unknown
    - Python dtype: `unknown`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RemoveAndInterpolateFramesNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "frames_to_drop": ("STRING", {"multiline": True, "default": "[8, 16, 24]"}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "replace_and_interpolate_frames"
    CATEGORY = "Steerable-Motion"

    def replace_and_interpolate_frames(self, images: torch.Tensor, frames_to_drop: str):
        if isinstance(frames_to_drop, str):
            frames_to_drop = eval(frames_to_drop)

        frames_to_drop = sorted(frames_to_drop, reverse=True)
        
        # Create instance of FILM_VFI within the function
        film_vfi = FILM_VFIImport()  # Assuming FILM_VFI does not require any special setup

        for index in frames_to_drop:
            if 0 < index < images.shape[0] - 1:
                # Extract the two surrounding frames
                batch = images[index-1:index+2:2]

                # Process through FILM_VFI
                interpolated_frames = film_vfi.vfi(
                    ckpt_name='film_net_fp32.pt', 
                    frames=batch, 
                    clear_cache_after_n_frames=10, 
                    multiplier=2
                )[0]  # Assuming vfi returns a tuple and the first element is the interpolated frames

                # Replace the original frames at the location
                images = torch.cat((images[:index-1], interpolated_frames, images[index+2:]))

        return (images,)

```
