---
tags:
- ControlNet
- Weight
---

# IPAdapter Weights
## Documentation
- Class name: `IPAdapterWeights`
- Category: `ipadapter/weights`
- Output node: `False`

The IPAdapterWeights node is designed to manage and manipulate weights within the IPAdapter framework, focusing on the application of weight strategies for image processing. It enables the customization of weight distribution across frames, supports various timing methods, and facilitates the integration of images with weight-based adjustments.
## Input types
### Required
- **`weights`**
    - A string of comma or newline-separated weight values to be converted into a list of floats. It forms the basis for weight distribution and manipulation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`timing`**
    - Specifies the timing strategy for applying weights across frames, influencing the animation or transition effect.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`frames`**
    - The total number of frames to which the weights will be applied, defining the scope of the animation or transition.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`start_frame`**
    - The starting frame number from which weight application begins, allowing for precise control over the animation's commencement.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_frame`**
    - The ending frame number at which weight application concludes, marking the animation's termination point.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`add_starting_frames`**
    - The number of additional frames to be added at the beginning of the animation, extending its start.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`add_ending_frames`**
    - The number of additional frames to be added at the end of the animation, extending its duration.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`method`**
    - The method used for applying weights across frames, affecting the pattern and distribution of weight adjustments.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`image`**
    - An optional image input that, if provided, can be used in conjunction with weights for enhanced image processing effects.
    - Comfy dtype: `IMAGE`
    - Python dtype: `str`
## Output types
- **`weights`**
    - Comfy dtype: `FLOAT`
    - The processed list of weight values after applying the specified strategy and adjustments.
    - Python dtype: `List[float]`
- **`weights_invert`**
    - Comfy dtype: `FLOAT`
    - The inverted list of weight values, providing an alternative distribution for certain processing effects.
    - Python dtype: `List[float]`
- **`total_frames`**
    - Comfy dtype: `INT`
    - The total number of frames resulting from the applied weight strategy, including any added starting or ending frames.
    - Python dtype: `int`
- **`image_1`**
    - Comfy dtype: `IMAGE`
    - The first image output, modified according to the weight strategy and method applied.
    - Python dtype: `str`
- **`image_2`**
    - Comfy dtype: `IMAGE`
    - The second image output, modified in relation to the first image and the applied weight strategy.
    - Python dtype: `str`
- **`weights_strategy`**
    - Comfy dtype: `WEIGHTS_STRATEGY`
    - The weight strategy applied, encapsulating all weight-related parameters and their effects on the output.
    - Python dtype: `Dict[str, Any]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class IPAdapterWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "weights": ("STRING", {"default": '1.0, 0.0', "multiline": True }),
            "timing": (["custom", "linear", "ease_in_out", "ease_in", "ease_out", "random"], { "default": "linear" } ),
            "frames": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1 }),
            "start_frame": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1 }),
            "end_frame": ("INT", {"default": 9999, "min": 0, "max": 9999, "step": 1 }),
            "add_starting_frames": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1 }),
            "add_ending_frames": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1 }),
            "method": (["full batch", "shift batches", "alternate batches"], { "default": "full batch" }),
            }, "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "INT", "IMAGE", "IMAGE", "WEIGHTS_STRATEGY")
    RETURN_NAMES = ("weights", "weights_invert", "total_frames", "image_1", "image_2", "weights_strategy")
    FUNCTION = "weights"
    CATEGORY = "ipadapter/weights"

    def weights(self, weights='', timing='custom', frames=0, start_frame=0, end_frame=9999, add_starting_frames=0, add_ending_frames=0, method='full batch', weights_strategy=None, image=None):
        import random

        frame_count = image.shape[0] if image is not None else 0
        if weights_strategy is not None:
            weights = weights_strategy["weights"]
            timing = weights_strategy["timing"]
            frames = weights_strategy["frames"]
            start_frame = weights_strategy["start_frame"]
            end_frame = weights_strategy["end_frame"]
            add_starting_frames = weights_strategy["add_starting_frames"]
            add_ending_frames = weights_strategy["add_ending_frames"]
            method = weights_strategy["method"]
            frame_count = weights_strategy["frame_count"]
        else:
            weights_strategy = {
                "weights": weights,
                "timing": timing,
                "frames": frames,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "add_starting_frames": add_starting_frames,
                "add_ending_frames": add_ending_frames,
                "method": method,
                "frame_count": frame_count,
            }

        # convert the string to a list of floats separated by commas or newlines
        weights = weights.replace("\n", ",")
        weights = [float(weight) for weight in weights.split(",") if weight.strip() != ""]

        if timing != "custom":
            frames = max(frames, 2)
            start = 0.0
            end = 1.0

            if len(weights) > 0:
                start = weights[0]
                end = weights[-1]

            weights = []

            end_frame = min(end_frame, frames)
            duration = end_frame - start_frame
            if start_frame > 0:
                weights.extend([start] * start_frame)

            for i in range(duration):
                n = duration - 1
                if timing == "linear":
                    weights.append(start + (end - start) * i / n)
                elif timing == "ease_in_out":
                    weights.append(start + (end - start) * (1 - math.cos(i / n * math.pi)) / 2)
                elif timing == "ease_in":
                    weights.append(start + (end - start) * math.sin(i / n * math.pi / 2))
                elif timing == "ease_out":
                    weights.append(start + (end - start) * (1 - math.cos(i / n * math.pi / 2)))
                elif timing == "random":
                    weights.append(random.uniform(start, end))

            weights[-1] = end if timing != "random" else weights[-1]
            if end_frame < frames:
                weights.extend([end] * (frames - end_frame))

        if len(weights) == 0:
            weights = [0.0]

        frames = len(weights)

        # repeat the images for cross fade
        image_1 = None
        image_2 = None

        # Calculate the min and max of the weights
        min_weight = min(weights)
        max_weight = max(weights)

        if image is not None:

            if "shift" in method:
                image_1 = image[:-1]
                image_2 = image[1:]

                weights = weights * image_1.shape[0]
                image_1 = image_1.repeat_interleave(frames, 0)
                image_2 = image_2.repeat_interleave(frames, 0)
            elif "alternate" in method:
                image_1 = image[::2].repeat_interleave(2, 0)
                image_1 = image_1[1:]
                image_2 = image[1::2].repeat_interleave(2, 0)

                # Invert the weights relative to their own range
                mew_weights = weights + [max_weight - (w - min_weight) for w in weights]

                mew_weights = mew_weights * (image_1.shape[0] // 2)
                if image.shape[0] % 2:
                    image_1 = image_1[:-1]
                else:
                    image_2 = image_2[:-1]
                    mew_weights = mew_weights + weights

                weights = mew_weights
                image_1 = image_1.repeat_interleave(frames, 0)
                image_2 = image_2.repeat_interleave(frames, 0)
            else:
                weights = weights * image.shape[0]
                image_1 = image.repeat_interleave(frames, 0)

            # add starting and ending frames
            if add_starting_frames > 0:
                weights = [weights[0]] * add_starting_frames + weights
                image_1 = torch.cat([image[:1].repeat(add_starting_frames, 1, 1, 1), image_1], dim=0)
                if image_2 is not None:
                    image_2 = torch.cat([image[:1].repeat(add_starting_frames, 1, 1, 1), image_2], dim=0)
            if add_ending_frames > 0:
                weights = weights + [weights[-1]] * add_ending_frames
                image_1 = torch.cat([image_1, image[-1:].repeat(add_ending_frames, 1, 1, 1)], dim=0)
                if image_2 is not None:
                    image_2 = torch.cat([image_2, image[-1:].repeat(add_ending_frames, 1, 1, 1)], dim=0)

        # reverse the weights array
        weights_invert = weights[::-1]

        frame_count = len(weights)

        return (weights, weights_invert, frame_count, image_1, image_2, weights_strategy,)

```
