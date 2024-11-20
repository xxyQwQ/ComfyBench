---
tags:
- Multimedia
- VideoHelperSuite
---

# Load Video (Upload) 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_LoadVideo`
- Category: `Video Helper Suite 🎥🅥🅗🅢`
- Output node: `False`

The VHS_LoadVideo node is designed to facilitate the loading of video files into a processing pipeline, handling tasks such as decoding video data, extracting frames, and optionally applying transformations like resizing or frame rate adjustment. It serves as a foundational component within the Video Helper Suite, enabling subsequent video analysis, manipulation, or enhancement operations.
## Input types
### Required
- **`video`**
    - Specifies the path to the video file to be loaded. It is crucial for locating and accessing the video data for processing.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`force_rate`**
    - Determines the frame rate to which the video should be adjusted during loading. This parameter allows for standardizing the frame rate across different videos.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`force_size`**
    - Indicates the desired resolution to which video frames should be resized. This can be essential for ensuring consistency in frame dimensions across various processing steps.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`custom_width`**
    - Specifies the custom width for resizing video frames. This parameter works in conjunction with custom_height to define the new dimensions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`custom_height`**
    - Specifies the custom height for resizing video frames, used together with custom_width to set the target frame size.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`frame_load_cap`**
    - Limits the number of frames to be loaded from the video, useful for processing long videos or for sampling purposes.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`skip_first_frames`**
    - Skips a specified number of frames at the beginning of the video. This can be useful for bypassing non-relevant content or intros.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`select_every_nth`**
    - Loads every Nth frame from the video, allowing for temporal downsampling and reducing the total number of frames processed.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`meta_batch`**
    - Defines a batch processing context for the video, enabling efficient handling of video frames in groups.
    - Comfy dtype: `VHS_BatchManager`
    - Python dtype: `object`
- **`vae`**
    - Optionally specifies a variational autoencoder model for processing the video frames, potentially for tasks like dimensionality reduction or feature extraction.
    - Comfy dtype: `VAE`
    - Python dtype: `object`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - The processed video frames, potentially resized or otherwise transformed, ready for further processing.
    - Python dtype: `List[np.ndarray]`
- **`frame_count`**
    - Comfy dtype: `INT`
    - The total number of frames loaded from the video, after applying any specified filters or limits.
    - Python dtype: `int`
- **`audio`**
    - Comfy dtype: `VHS_AUDIO`
    - The extracted audio track from the video, if applicable and requested.
    - Python dtype: `List[np.ndarray]`
- **`video_info`**
    - Comfy dtype: `VHS_VIDEOINFO`
    - Metadata about the video and the loading process, including original and loaded frame rates, dimensions, and duration.
    - Python dtype: `Dict[str, Any]`
- **`LATENT`**
    - Comfy dtype: `LATENT`
    - The representation of video frames in a latent space, if a variational autoencoder (VAE) is used during the loading process.
    - Python dtype: `Dict[str, List[np.ndarray]]`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - [VHS_VideoCombine](../../ComfyUI-VideoHelperSuite/Nodes/VHS_VideoCombine.md)
    - [VAEEncode](../../Comfy/Nodes/VAEEncode.md)
    - [EmptyLatentImage](../../Comfy/Nodes/EmptyLatentImage.md)
    - [CannyEdgePreprocessor](../../comfyui_controlnet_aux/Nodes/CannyEdgePreprocessor.md)
    - [ImageScaleToTotalPixels](../../Comfy/Nodes/ImageScaleToTotalPixels.md)
    - [ImageResize+](../../ComfyUI_essentials/Nodes/ImageResize+.md)
    - [TilePreprocessor](../../comfyui_controlnet_aux/Nodes/TilePreprocessor.md)
    - [LineArtPreprocessor](../../comfyui_controlnet_aux/Nodes/LineArtPreprocessor.md)
    - [DWPreprocessor](../../comfyui_controlnet_aux/Nodes/DWPreprocessor.md)



## Source code
```python
class LoadVideoUpload:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1] in video_extensions):
                    files.append(f)
        return {"required": {
                    "video": (sorted(files),),
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
        kwargs['video'] = folder_paths.get_annotated_filepath(strip_path(kwargs['video']))
        return load_video_cv(**kwargs)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        image_path = folder_paths.get_annotated_filepath(video)
        return calculate_file_hash(image_path)

    @classmethod
    def VALIDATE_INPUTS(s, video, force_size, **kwargs):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid video file: {}".format(video)
        return True

```
