---
tags:
- Image
- ImageBlend
- ImageComposite
---

# Image Batch Filmic Transitions
## Documentation
- Class name: `SaltFilmicTransitions`
- Category: `SALT/AudioViz/Scheduling/Image`
- Output node: `False`

This node specializes in generating filmic transitions between two sets of images, offering a variety of modes such as swipes and circle expansions. It allows for the creation of smooth, visually appealing transitions with customizable frame counts and optional mask blur effects, enhancing the dynamic storytelling capabilities of image sequences.
## Input types
### Required
- **`images_a`**
    - The first set of images to transition from. These images serve as the starting point for the filmic transition effect.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`images_b`**
    - The second set of images to transition to. These images serve as the destination for the filmic transition effect.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`mode`**
    - Specifies the type of transition effect to apply, such as swipe directions or circle expansions. This choice determines the visual style of the transition between the two image sets.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`transition_frames`**
    - The number of frames to use for the transition, allowing control over the speed and smoothness of the effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`mask_blur_schedule`**
    - An optional schedule for blurring the mask used in the transition, enabling dynamic blur effects throughout the transition process.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
## Output types
- **`images`**
    - Comfy dtype: `IMAGE`
    - The resulting set of images after applying the filmic transition, combined into a single tensor.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SaltFilmicTransitions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_a": ("IMAGE", ),
                "images_b": ("IMAGE", ),
                "mode": (["sipe_left", "swipe_right", "swipe_up", "swipe_down", "diagonal_tl_br", "diagonal_tr_bl", "diagonal_bl_tr", "diagonal_br_tl", "circle_expand", "circle_contract"],),
                "transition_frames": ("INT", {"min": 2, "max": 1024, "default": 10}),
            },
            "optional": {
                "mask_blur_schedule": ("LIST", ),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "generate_transition"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Image"

    def generate_transition(self, images_a, images_b, mode, transition_frames, mask_blur_schedule=[0]):
        mask_blur_schedule = [float(val) for val in mask_blur_schedule]
        img_list_a = [tensor2pil(img) for img in images_a]
        img_list_b = [tensor2pil(img) for img in images_b]
        transition = ImageBatchTransition(img_list_a, img_list_b, frames_per_transition=int(transition_frames), blur_radius=mask_blur_schedule, mode=mode)
        result_images = transition.create_transition()
        result_images = [pil2tensor(img) for img in result_images]
        return (torch.cat(result_images, dim=0), )

```
