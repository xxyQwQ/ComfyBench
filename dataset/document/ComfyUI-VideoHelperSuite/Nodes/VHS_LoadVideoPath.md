---
tags:
- Multimedia
- VideoHelperSuite
---

# Load Video (Path) 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_LoadVideoPath`
- Category: `Video Helper Suite 🎥🅥🅗🅢`
- Output node: `False`

This node is designed to load video files from a specified path, applying various preprocessing steps such as resizing, frame rate adjustment, and optional VAE encoding. It facilitates the extraction and manipulation of video data for further processing or analysis within the Video Helper Suite.
## Input types
### Required
- **`video`**
    - The path to the video file to be loaded. It is crucial for locating and accessing the video data for processing.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`force_rate`**
    - Specifies the target frame rate to which the video should be adjusted. This parameter is essential for standardizing the frame rate across different videos.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`force_size`**
    - Defines the target resolution for the video. This parameter is key in resizing the video to a specific dimension.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list of str`
- **`custom_width`**
    - The custom width to which the video should be resized. This allows for precise control over the video's width dimension.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`custom_height`**
    - The custom height to which the video should be resized. This allows for precise control over the video's height dimension.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`frame_load_cap`**
    - Limits the number of frames to be loaded from the video. This parameter helps in managing memory usage by restricting the frame count.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`skip_first_frames`**
    - The number of initial frames to skip. This is useful for bypassing unneeded content at the beginning of the video.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`select_every_nth`**
    - Determines the interval at which frames are selected. By choosing every nth frame, it reduces the total number of frames processed and can help in focusing on specific segments of the video.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`meta_batch`**
    - unknown
    - Comfy dtype: `VHS_BatchManager`
    - Python dtype: `unknown`
- **`vae`**
    - unknown
    - Comfy dtype: `VAE`
    - Python dtype: `unknown`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - The processed frames of the video, potentially resized and adjusted according to the specified frame rate and dimensions.
    - Python dtype: `List[torch.Tensor]`
- **`frame_count`**
    - Comfy dtype: `INT`
    - The total number of frames loaded and processed from the video.
    - Python dtype: `int`
- **`audio`**
    - Comfy dtype: `VHS_AUDIO`
    - The extracted audio track from the video, adjusted according to the specified frame selection parameters.
    - Python dtype: `torch.Tensor`
- **`video_info`**
    - Comfy dtype: `VHS_VIDEOINFO`
    - Metadata about the video, including both source and loaded properties such as frame rate, duration, and dimensions.
    - Python dtype: `Dict[str, Any]`
- **`LATENT`**
    - Comfy dtype: `LATENT`
    - The optional VAE-encoded representation of the video frames, if a VAE model is applied.
    - Python dtype: `Optional[Dict[str, List[torch.Tensor]]]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LoadVideoPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("STRING", {"default": "X://insert/path/here.mp4", "vhs_path_extensions": video_extensions}),
                "force_rate": ("INT", {"default": 0, "min": 0, "max": 60, "step": 1}),
                 "force_size": (["Disabled", "Custom Height", "Custom Width", "Custom", "256x?", "?x256", "256x256", "512x?", "?x512", "512x512"],),
                 "custom_width": ("INT", {"default": 512, "min": 0, "max": DIMMAX, "step": 8}),
                 "custom_height": ("INT", {"default": 512, "min": 0, "max": DIMMAX, "step": 8}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = ("IMAGE", "INT", "VHS_AUDIO", "VHS_VIDEOINFO", "LATENT")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info", "LATENT")

    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        if kwargs['video'] is None or validate_path(kwargs['video']) != True:
            raise Exception("video is not a valid path: " + kwargs['video'])
        return load_video_cv(**kwargs)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        return hash_path(video)

    @classmethod
    def VALIDATE_INPUTS(s, video, **kwargs):
        return validate_path(video, allow_none=True)

```
