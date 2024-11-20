---
tags:
- Multimedia
- VideoHelperSuite
---

# Load Video From Url
## Documentation
- Class name: `LoadVideoFromUrl`
- Category: `Art Venture/Loaders`
- Output node: `False`

The LoadVideoFromUrl node is designed to facilitate the loading of video content from URLs into a usable format for further processing or analysis. It supports handling various URL schemes, including direct links to video files, local file paths, and specialized URLs with parameters for video retrieval. This node also includes functionality to manage video files by downloading or relocating them as needed, ensuring compatibility with the system's file structure.
## Input types
### Required
- **`video`**
    - The 'video' parameter is the URL or path of the video to be loaded. It plays a crucial role in determining the source from which the video will be fetched or loaded, directly influencing the node's processing logic and output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`force_rate`**
    - This parameter allows for the specification of a forced frame rate for the loaded video, affecting the temporal resolution and playback speed.
    - Comfy dtype: `INT`
    - Python dtype: `Optional[float]`
- **`force_size`**
    - Allows setting a forced resolution for the loaded video, which can alter the spatial dimensions of the output frames.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `Optional[str]`
- **`custom_width`**
    - Specifies a custom width for the output video frames, overriding the default or source video's width.
    - Comfy dtype: `INT`
    - Python dtype: `Optional[int]`
- **`custom_height`**
    - Specifies a custom height for the output video frames, overriding the default or source video's height.
    - Comfy dtype: `INT`
    - Python dtype: `Optional[int]`
- **`frame_load_cap`**
    - Limits the number of frames to be loaded from the video, useful for reducing memory usage or focusing on specific segments.
    - Comfy dtype: `INT`
    - Python dtype: `Optional[int]`
- **`skip_first_frames`**
    - Skips a specified number of frames from the beginning of the video, useful for bypassing unneeded content or intros.
    - Comfy dtype: `INT`
    - Python dtype: `Optional[int]`
- **`select_every_nth`**
    - Loads every nth frame from the video, allowing for temporal downsampling and reduced dataset size.
    - Comfy dtype: `INT`
    - Python dtype: `Optional[int]`
### Optional
- **`meta_batch`**
    - A parameter for batch processing metadata, potentially used for optimizing load operations or handling multiple videos.
    - Comfy dtype: `VHS_BatchManager`
    - Python dtype: `Optional[dict]`
- **`vae`**
    - An optional VAE model parameter for potential preprocessing or encoding of video frames.
    - Comfy dtype: `VAE`
    - Python dtype: `Optional[torch.nn.Module]`
## Output types
- **`frames`**
    - Comfy dtype: `IMAGE`
    - A list of loaded videos, each represented as a tensor. This output is crucial for subsequent video processing tasks.
    - Python dtype: `List[torch.Tensor]`
- **`frame_count`**
    - Comfy dtype: `INT`
    - A list containing the number of frames for each video loaded. This information is essential for frame-based video analysis and processing.
    - Python dtype: `List[int]`
- **`has_video`**
    - Comfy dtype: `BOOLEAN`
    - A boolean indicating whether any video was successfully loaded. This flag can be used to trigger further processing or handle errors.
    - Python dtype: `bool`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
    class UtilLoadVideoFromUrl(LoadVideoPath):
        @classmethod
        def INPUT_TYPES(s):
            inputs = LoadVideoPath.INPUT_TYPES()
            inputs["required"]["video"] = ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False})
            return inputs

        CATEGORY = "Art Venture/Loaders"
        FUNCTION = "load"
        RETURN_TYPES = ("IMAGE", "INT", "BOOLEAN")
        RETURN_NAMES = ("frames", "frame_count", "has_video")
        OUTPUT_IS_LIST = (True, True, False)

        def load_gif(
            self,
            gif_path: str,
            force_rate: int,
            force_size: str,
            skip_first_frames: int,
            frame_load_cap: int,
            select_every_nth: int,
        ):
            from PIL import Image, ImageSequence

            image = Image.open(gif_path)
            frames = []
            total_frames_evaluated = -1

            if force_rate != 0:
                print(f"Force rate is not supported for gifs/webps")
            if frame_load_cap == 0:
                frame_load_cap = 999999999

            for i, frame in enumerate(ImageSequence.Iterator(image)):
                if i < skip_first_frames:
                    continue
                elif i >= skip_first_frames + frame_load_cap:
                    break
                else:
                    total_frames_evaluated += 1
                    if total_frames_evaluated % select_every_nth == 0:
                        frames.append(pil2tensor(frame.copy().convert("RGB")))

            images = torch.cat(frames, dim=0)

            if force_size != "Disabled":
                height = images.shape[1]
                width = images.shape[2]
                new_size = target_size(width, height, force_size)
                if new_size[0] != width or new_size[1] != height:
                    s = images.movedim(-1, 1)
                    s = common_upscale(s, new_size[0], new_size[1], "lanczos", "disabled")
                    images = s.movedim(1, -1)

            return (images, len(frames))

        def load_url(self, video: str, **kwargs):
            url = video.strip('"')

            if url == "":
                return (None, 0)

            if os.path.isfile(url):
                pass
            elif url.startswith("file://"):
                url = url[7:]
                url = os.path.abspath(url)

                if not os.path.isfile(url):
                    raise Exception(f"File {url} does not exist")

                if url.startswith(input_dir):
                    video = url[len(input_dir) + 1 :] + " [input]"
                elif url.startswith(output_dir):
                    video = url[len(output_dir) + 1 :] + " [output]"
                elif url.startswith(temp_dir):
                    video = url[len(temp_dir) + 1 :] + " [temp]"
                else:
                    # move file to temp_dir
                    import shutil

                    tempdir = os.path.join(temp_dir, "video")
                    if not os.path.exists(tempdir):
                        os.makedirs(tempfile, exist_ok=True)

                    filename = os.path.basename(url)
                    filepath = os.path.join(tempdir, filename)

                    i = 1
                    split = os.path.splitext(filename)
                    while os.path.exists(filepath):
                        filename = f"{split[0]} ({i}){split[1]}"
                        filepath = os.path.join(tempdir, filename)
                        i += 1

                    shutil.copy(url, filepath)
                    video = "video/" + filename + " [temp]"
            elif url.startswith("http://") or url.startswith("https://"):
                from torch.hub import download_url_to_file
                from urllib.parse import urlparse

                parts = urlparse(url)
                filename = os.path.basename(parts.path)
                tempfile = os.path.join(temp_dir, "video")
                if not os.path.exists(tempfile):
                    os.makedirs(tempfile, exist_ok=True)
                tempfile = os.path.join(tempfile, filename)

                print(f'Downloading: "{url}" to {tempfile}\n')
                download_url_to_file(url, tempfile, progress=True)

                video = "video/" + filename + " [temp]"
            elif url.startswith("/view?"):
                from urllib.parse import parse_qs

                qs = parse_qs(url[6:])
                filename = qs.get("name", qs.get("filename", None))
                if filename is None:
                    raise Exception(f"Invalid url: {url}")

                filename = filename[0]
                subfolder = qs.get("subfolder", None)
                if subfolder is not None:
                    filename = os.path.join(subfolder[0], filename)

                dirtype = qs.get("type", ["input"])
                video = f"{filename} [{dirtype[0]}]"
            else:
                raise Exception(f"Invalid url: {url}")

            if ".gif [" in video.lower() or ".webp [" in video.lower():
                gif_path = folder_paths.get_annotated_filepath(video.strip('"'))
                res = self.load_gif(gif_path, **kwargs)
            else:
                res = self.load_video(video=video, **kwargs)

            return res

        def load(self, video: str, **kwargs):
            urls = video.strip().split("\n")

            videos = []
            frame_counts = []

            for url in urls:
                images, frame_count = self.load_url(url, **kwargs)
                if images is not None and frame_count > 0:
                    videos.append(images)
                    frame_counts.append(frame_count)

            has_video = len(videos) > 0
            if not has_video:
                image = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")
                videos.append(image)
                frame_counts.append(1)

            return (videos, frame_counts, has_video)

        @classmethod
        def IS_CHANGED(s, video: str, **kwargs):
            return video

        @classmethod
        def VALIDATE_INPUTS(s, **kwargs):
            return True

```
