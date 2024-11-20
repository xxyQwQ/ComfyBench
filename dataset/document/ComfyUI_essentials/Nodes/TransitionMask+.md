---
tags:
- Mask
- MaskGeneration
---

# 🔧 Transition Mask
## Documentation
- Class name: `TransitionMask+`
- Category: `essentials/mask`
- Output node: `False`

The TransitionMask+ node specializes in generating dynamic transition effects within masks, offering a range of transition styles and timing functions to create complex visual transitions between frames or states. This node simplifies the creation of animated or static transition effects, making it easier for users to apply sophisticated mask transitions in their projects.
## Input types
### Required
- **`width`**
    - Defines the width of the mask to be generated, setting the horizontal dimension of the transition effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Sets the height of the mask, determining the vertical dimension of the transition effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`frames`**
    - Specifies the total number of frames in the transition animation, controlling the length of the transition effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`start_frame`**
    - Indicates the starting frame number for the transition effect, allowing for control over the animation's beginning.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_frame`**
    - Determines the ending frame number for the transition, enabling customization of the animation's duration.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`transition_type`**
    - Selects the type of transition effect to be applied, such as slides, bars, boxes, or fades, offering a variety of visual styles.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`timing_function`**
    - Chooses the timing function for the transition effect, such as linear or ease-in-out, affecting the pacing of the transition.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - Produces a mask that represents the transition effect, which can be used to apply or visualize the transition within an image or a series of images.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class TransitionMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", { "default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1, }),
                "frames": ("INT", { "default": 16, "min": 1, "max": 9999, "step": 1, }),
                "start_frame": ("INT", { "default": 0, "min": 0, "step": 1, }),
                "end_frame": ("INT", { "default": 9999, "min": 0, "step": 1, }),
                "transition_type": (["horizontal slide", "vertical slide", "horizontal bar", "vertical bar", "center box", "horizontal door", "vertical door", "circle", "fade"],),
                "timing_function": (["linear", "in", "out", "in-out"],)
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def linear(self, i, t):
        return i/t
    def ease_in(self, i, t):
        return pow(i/t, 2)
    def ease_out(self, i, t):
        return 1 - pow(1 - i/t, 2)
    def ease_in_out(self, i, t):
        if i < t/2:
            return pow(i/(t/2), 2) / 2
        else:
            return 1 - pow(1 - (i - t/2)/(t/2), 2) / 2

    def execute(self, width, height, frames, start_frame, end_frame, transition_type, timing_function):
        if timing_function == 'in':
            timing_function = self.ease_in
        elif timing_function == 'out':
            timing_function = self.ease_out
        elif timing_function == 'in-out':
            timing_function = self.ease_in_out
        else:
            timing_function = self.linear

        out = []

        end_frame = min(frames, end_frame)
        transition = end_frame - start_frame

        if start_frame > 0:
            out = out + [torch.full((height, width), 0.0, dtype=torch.float32, device="cpu")] * start_frame

        for i in range(transition):
            frame = torch.full((height, width), 0.0, dtype=torch.float32, device="cpu")
            progress = timing_function(i, transition-1)

            if "horizontal slide" in transition_type:
                pos = round(width*progress)
                frame[:, :pos] = 1.0
            elif "vertical slide" in transition_type:
                pos = round(height*progress)
                frame[:pos, :] = 1.0
            elif "box" in transition_type:
                box_w = round(width*progress)
                box_h = round(height*progress)
                x1 = (width - box_w) // 2
                y1 = (height - box_h) // 2
                x2 = x1 + box_w
                y2 = y1 + box_h
                frame[y1:y2, x1:x2] = 1.0
            elif "circle" in transition_type:
                radius = math.ceil(math.sqrt(pow(width,2)+pow(height,2))*progress/2)
                c_x = width // 2
                c_y = height // 2
                # is this real life? Am I hallucinating?
                x = torch.arange(0, width, dtype=torch.float32, device="cpu")
                y = torch.arange(0, height, dtype=torch.float32, device="cpu")
                y, x = torch.meshgrid((y, x), indexing="ij")
                circle = ((x - c_x) ** 2 + (y - c_y) ** 2) <= (radius ** 2)
                frame[circle] = 1.0
            elif "horizontal bar" in transition_type:
                bar = round(height*progress)
                y1 = (height - bar) // 2
                y2 = y1 + bar
                frame[y1:y2, :] = 1.0
            elif "vertical bar" in transition_type:
                bar = round(width*progress)
                x1 = (width - bar) // 2
                x2 = x1 + bar
                frame[:, x1:x2] = 1.0
            elif "horizontal door" in transition_type:
                bar = math.ceil(height*progress/2)
                if bar > 0:
                    frame[:bar, :] = 1.0
                    frame[-bar:, :] = 1.0
            elif "vertical door" in transition_type:
                bar = math.ceil(width*progress/2)
                if bar > 0:
                    frame[:, :bar] = 1.0
                    frame[:, -bar:] = 1.0
            elif "fade" in transition_type:
                frame[:,:] = progress

            out.append(frame)

        if end_frame < frames:
            out = out + [torch.full((height, width), 1.0, dtype=torch.float32, device="cpu")] * (frames - end_frame)

        out = torch.stack(out, dim=0)

        return (out, )

```
