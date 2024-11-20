---
tags:
- Image
- ImageSequence
---

# Load Image Sequence (mtb)
## Documentation
- Class name: `Load Image Sequence (mtb)`
- Category: `mtb/IO`
- Output node: `False`

The Load Image Sequence node is designed for loading sequences of images from a specified directory, allowing for the dynamic retrieval of individual frames or batches of frames based on the current frame index or a specified range. It supports loading all frames at once or fetching specific frames, making it versatile for applications requiring sequential image processing or animation.
## Input types
### Required
- **`path`**
    - Specifies the directory path where the image sequence is stored. This path is crucial for locating and loading the desired images.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`current_frame`**
    - Determines the specific frame to load from the sequence. Setting this to -1 instructs the node to load all frames available, facilitating batch processing or full sequence retrieval.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`range`**
    - Defines a specific range of frames to load from the sequence, offering fine-grained control over which images are retrieved for processing.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The loaded image or images from the specified sequence. This output can be a single frame or a batch of frames, depending on the input parameters.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The corresponding mask or masks for the loaded image(s), providing additional data for processing or analysis.
    - Python dtype: `torch.Tensor`
- **`current_frame`**
    - Comfy dtype: `INT`
    - The index of the current frame that was loaded, useful for tracking progress through a sequence.
    - Python dtype: `int`
- **`total_frames`**
    - Comfy dtype: `INT`
    - The total number of frames loaded, which can be used to understand the scope of the sequence being processed.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MTB_LoadImageSequence:
    """Load an image sequence from a folder. The current frame is used to determine which image to load.

    Usually used in conjunction with the `Primitive` node set to increment to load a sequence of images from a folder.
    Use -1 to load all matching frames as a batch.

    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "videos/####.png"}),
                "current_frame": (
                    "INT",
                    {"default": 0, "min": -1, "max": 9999999},
                ),
            },
            "optional": {
                "range": ("STRING", {"default": ""}),
            },
        }

    CATEGORY = "mtb/IO"
    FUNCTION = "load_image"
    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "image",
        "mask",
        "current_frame",
        "total_frames",
    )

    def load_image(self, path=None, current_frame=0, range=""):
        load_all = current_frame == -1
        total_frames = 1

        if range:
            frames = self.get_frames_from_range(path, range)
            imgs, masks = zip(*(img_from_path(frame) for frame in frames))
            out_img = torch.cat(imgs, dim=0)
            out_mask = torch.cat(masks, dim=0)
            total_frames = len(imgs)
            return (out_img, out_mask, -1, total_frames)

        elif load_all:
            log.debug(f"Loading all frames from {path}")
            frames = resolve_all_frames(path)
            log.debug(f"Found {len(frames)} frames")

            imgs = []
            masks = []

            imgs, masks = zip(*(img_from_path(frame) for frame in frames))

            out_img = torch.cat(imgs, dim=0)
            out_mask = torch.cat(masks, dim=0)
            total_frames = len(imgs)

            return (out_img, out_mask, -1, total_frames)

        log.debug(f"Loading image: {path}, {current_frame}")
        resolved_path = resolve_path(path, current_frame)
        image_path = folder_paths.get_annotated_filepath(resolved_path)
        image, mask = img_from_path(image_path)
        return (image, mask, current_frame, total_frames)

    def get_frames_from_range(self, path, range_str):
        try:
            start, end = map(int, range_str.split("-"))
        except ValueError:
            raise ValueError(
                f"Invalid range format: {range_str}. Expected format is 'start-end'."
            )

        frames = resolve_all_frames(path)
        total_frames = len(frames)

        if start < 0 or end >= total_frames:
            raise ValueError(
                f"Range {range_str} is out of bounds. Total frames available: {total_frames}"
            )

        if "#" in path:
            frame_regex = re.escape(path).replace(r"\#", r"(\d+)")
            frame_number_regex = re.compile(frame_regex)

            matching_frames = []
            for frame in frames:
                match = frame_number_regex.search(frame)

                if match:
                    frame_number = int(match.group(1))
                    if start <= frame_number <= end:
                        matching_frames.append(frame)

            return matching_frames
        else:
            log.warning(
                f"Wildcard pattern or directory will use indexes instead of frame numbers for : {path}"
            )

            selected_frames = frames[start : end + 1]

        return selected_frames

    @staticmethod
    def IS_CHANGED(path="", current_frame=0, range=""):
        print(f"Checking if changed: {path}, {current_frame}")
        if range or current_frame == -1:
            resolved_paths = resolve_all_frames(path)
            timestamps = [
                os.path.getmtime(folder_paths.get_annotated_filepath(p))
                for p in resolved_paths
            ]
            combined_hash = hashlib.sha256(
                "".join(map(str, timestamps)).encode()
            )
            return combined_hash.hexdigest()
        resolved_path = resolve_path(path, current_frame)
        image_path = folder_paths.get_annotated_filepath(resolved_path)
        if os.path.exists(image_path):
            m = hashlib.sha256()
            with open(image_path, "rb") as f:
                m.update(f.read())
            return m.digest().hex()
        return "NONE"

    # @staticmethod
    # def VALIDATE_INPUTS(path="", current_frame=0):

    #     print(f"Validating inputs: {path}, {current_frame}")
    #     resolved_path = resolve_path(path, current_frame)
    #     if not folder_paths.exists_annotated_filepath(resolved_path):
    #         return f"Invalid image file: {resolved_path}"
    #     return True

```
