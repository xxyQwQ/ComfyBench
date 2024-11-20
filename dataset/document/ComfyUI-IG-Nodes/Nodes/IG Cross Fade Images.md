---
tags:
- AnimationScheduling
- VisualEffects
---

# 🧑🏻‍🧑🏿‍🧒🏽 IG Cross Fade Images
## Documentation
- Class name: `IG Cross Fade Images`
- Category: `🐓 IG Nodes/Interpolation`
- Output node: `False`

The IG Cross Fade Images node is designed for creating a smooth transition between a sequence of images. It leverages crossfading techniques combined with easing functions to interpolate between images, allowing for customizable transitions in terms of duration and style. This node is particularly useful in generating animations or video effects where a seamless blend from one image to another is desired.
## Input types
### Required
- **`input_images`**
    - A list of image tensors to be crossfaded. This parameter is crucial for defining the sequence of images that will undergo the transition process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`interpolation`**
    - Determines the easing function applied to the transition, influencing the animation's dynamic and feel.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`transitioning_frames`**
    - Specifies the number of frames to be used for transitioning between each pair of images, affecting the smoothness and duration of the crossfade effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`repeat_count`**
    - Controls how many times each image is repeated before transitioning to the next, allowing for extended display of certain frames within the sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - A tensor containing the sequence of crossfaded images, ready for visualization or further processing.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class IG_CrossFadeImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "input_images": ("IMAGE",),
                 "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out", "bounce", "elastic", "glitchy", "exponential_ease_out"],),
                 "transitioning_frames": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
                 "repeat_count": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
        }
    } 
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = TREE_INTERP

    def main(self, input_images, transitioning_frames, interpolation, repeat_count):

        # Assuming input_images is a list of tensors with shape [C, H, W]
        # Initialize an empty list to hold crossfaded images
        crossfade_images = []
        image_count = len(input_images)

        for i in range(image_count - 1):  # For each pair of images
            image1 = input_images[i]
            image2 = input_images[i + 1]
            for repeat in range(repeat_count - transitioning_frames):  # Repeat the current image
                crossfade_images.append(image1)
            alphas = torch.linspace(1.0 / (transitioning_frames + 1.0), 1.0 - 1.0 / (transitioning_frames + 1.0), transitioning_frames + 1)
            for alpha in alphas:  # Transition to the next image
                easing_function = easing_functions[interpolation]
                eased_alpha = easing_function(alpha.item())
                crossfaded_image = crossfade(image1, image2, eased_alpha)
                crossfade_images.append(crossfaded_image)

        # Handle the last image repetition
        for repeat in range(repeat_count):
            crossfade_images.append(input_images[-1])
        # crossfade_images.append(last_image)

        crossfade_images = torch.stack(crossfade_images, dim=0)
    
        # If not at end, transition image
            

        # for i in range(transitioning_frames):
        #     alpha = alphas[i]
        #     image1 = images_1[i + transition_start_index]
        #     image2 = images_2[i + transition_start_index]
        #     easing_function = easing_functions.get(interpolation)
        #     alpha = easing_function(alpha)  # Apply the easing function to the alpha value

        #     crossfade_image = crossfade(image1, image2, alpha)
        #     crossfade_images.append(crossfade_image)
            
        # # Convert crossfade_images to tensor
        # crossfade_images = torch.stack(crossfade_images, dim=0)
        # # Get the last frame result of the interpolation
        # last_frame = crossfade_images[-1]
        # # Calculate the number of remaining frames from images_2
        # remaining_frames = len(images_2) - (transition_start_index + transitioning_frames)
        # # Crossfade the remaining frames with the last used alpha value
        # for i in range(remaining_frames):
        #     alpha = alphas[-1]
        #     image1 = images_1[i + transition_start_index + transitioning_frames]
        #     image2 = images_2[i + transition_start_index + transitioning_frames]
        #     easing_function = easing_functions.get(interpolation)
        #     alpha = easing_function(alpha)  # Apply the easing function to the alpha value

        #     crossfade_image = crossfade(image1, image2, alpha)
        #     crossfade_images = torch.cat([crossfade_images, crossfade_image.unsqueeze(0)], dim=0)
        # # Append the beginning of images_1
        # beginning_images_1 = images_1[:transition_start_index]
        # crossfade_images = torch.cat([beginning_images_1, crossfade_images], dim=0)
        return (crossfade_images, )

```
