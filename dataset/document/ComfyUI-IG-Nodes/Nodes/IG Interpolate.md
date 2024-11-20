# 🧑🏻‍🧑🏿‍🧒🏽 IG Interpolate
## Documentation
- Class name: `IG Interpolate`
- Category: `🐓 IG Nodes/Interpolation`
- Output node: `False`

The IG Interpolate node is designed to facilitate the exploration of latent space by interpolating between embeddings of input images. It generates a sequence of in-between embeddings that smoothly transition from one image to the next, allowing for the creation of intermediate images that blend the features of the input images. This node is particularly useful for visualizing the path between points in latent space and understanding the underlying structure of the space.
## Input types
### Required
- **`ipadapter`**
    - The ipadapter parameter is an interface for integrating with external image processing adapters, essential for configuring the node's interaction with different image processing models.
    - Comfy dtype: `IPADAPTER`
    - Python dtype: `object`
- **`clip_vision`**
    - The clip_vision parameter specifies the CLIP vision model to be used for encoding the input images into embeddings. This model plays a critical role in the interpolation process by providing the necessary embeddings for the input images.
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `object`
- **`transitioning_frames`**
    - The transitioning_frames parameter determines the number of frames to be generated between each pair of input images, directly affecting the smoothness and length of the interpolation sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`repeat_count`**
    - The repeat_count parameter controls the number of times each interpolated embedding is repeated in the output sequence, allowing for the adjustment of the pacing and duration of the visual transition between images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`interpolation`**
    - The interpolation parameter selects the method of interpolation (e.g., linear, ease_in, etc.) to be used for transitioning between embeddings, influencing the dynamics of the transition.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`buffer`**
    - The buffer parameter specifies the number of repetitions for the first and last embeddings in the sequence, creating a 'buffer' period at the start and end of the interpolation. This helps in creating smoother transitions and more coherent visualizations of the interpolation path.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`input_images1`**
    - The input_images1 parameter is the first set of images for which embeddings are to be interpolated, serving as one of the starting points for the interpolation process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`input_images2`**
    - The input_images2 parameter is the second set of images for which embeddings are to be interpolated, serving as another starting point for the interpolation process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`input_images3`**
    - The input_images3 parameter is the third set of images for which embeddings are to be interpolated, providing additional starting points for the interpolation process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`positive_prompts`**
    - The positive_prompts parameter is a collection of text prompts that describe desired attributes or features to be emphasized in the interpolated images.
    - Comfy dtype: `STRING`
    - Python dtype: `list[str]`
- **`negative_prompts`**
    - The negative_prompts parameter is a collection of text prompts that describe undesired attributes or features to be minimized in the interpolated images.
    - Comfy dtype: `STRING`
    - Python dtype: `list[str]`
## Output types
- **`pos_embeds1`**
    - Comfy dtype: `EMBEDS`
    - The pos_embeds1 output represents the interpolated embeddings corresponding to the first set of input images.
    - Python dtype: `torch.Tensor`
- **`pos_embeds2`**
    - Comfy dtype: `EMBEDS`
    - The pos_embeds2 output represents the interpolated embeddings corresponding to the second set of input images.
    - Python dtype: `torch.Tensor`
- **`pos_embeds3`**
    - Comfy dtype: `EMBEDS`
    - The pos_embeds3 output represents the interpolated embeddings corresponding to the third set of input images.
    - Python dtype: `torch.Tensor`
- **`neg_embeds`**
    - Comfy dtype: `EMBEDS`
    - The neg_embeds output represents the embeddings generated without conditioning on the input images, used for comparison or as a baseline in the interpolation process.
    - Python dtype: `torch.Tensor`
- **`positive_string`**
    - Comfy dtype: `STRING`
    - The positive_string output is a formatted string of positive prompts, structured to facilitate their integration into the interpolation process.
    - Python dtype: `str`
- **`negative_string`**
    - Comfy dtype: `STRING`
    - The negative_string output is a formatted string of negative prompts, structured to facilitate their integration into the interpolation process.
    - Python dtype: `str`
- **`ipadapter`**
    - Comfy dtype: `IPADAPTER`
    - The ipadapter output returns the configured image processing adapter used in the node, providing context for the interpolation process.
    - Python dtype: `object`
- **`BATCH_SIZE`**
    - Comfy dtype: `INT`
    - The BATCH_SIZE output indicates the total number of embeddings generated during the interpolation process, reflecting the scale of the operation.
    - Python dtype: `int`
- **`FRAMES_TO_DROP`**
    - Comfy dtype: `STRING`
    - The FRAMES_TO_DROP output lists specific frames that should be omitted from the final interpolated sequence to optimize the visual transition.
    - Python dtype: `list[int]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class IG_Interpolate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": ("IPADAPTER", ),
                "clip_vision": ("CLIP_VISION",),
                "transitioning_frames": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
                "repeat_count": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
                "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out", "bounce", "elastic", "glitchy", "exponential_ease_out"],),
                "buffer": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            }, 
            "optional": {
                "input_images1": ("IMAGE",),
                "input_images2": ("IMAGE",),
                "input_images3": ("IMAGE",),
                "positive_prompts": ("STRING", {"default": [], "forceInput": True}),
                "negative_prompts": ("STRING", {"default": [], "forceInput": True}),
            }
        }

    RETURN_TYPES = ("EMBEDS", "EMBEDS", "EMBEDS", "EMBEDS", "STRING", "STRING", "IPADAPTER", "INT", "STRING",)
    RETURN_NAMES = ("pos_embeds1", "pos_embeds2", "pos_embeds3", "neg_embeds", "positive_string", "negative_string", "ipadapter", "BATCH_SIZE", "FRAMES_TO_DROP",)
    FUNCTION = "main"
    CATEGORY = TREE_INTERP 

    def main(self, ipadapter, clip_vision, transitioning_frames, repeat_count, interpolation, buffer, input_images1=None, input_images2=None, input_images3=None, positive_prompts=None, negative_prompts=None):
        if 'ipadapter' in ipadapter:
            ipadapter_model = ipadapter['ipadapter']['model']
            clip_vision = clip_vision if clip_vision is not None else ipadapter['clipvision']['model']
        else:
            ipadapter_model = ipadapter
            clip_vision = clip_vision
        if clip_vision is None:
            raise Exception("Missing CLIPVision model.")
        
        is_plus = "proj.3.weight" in ipadapter_model["image_proj"] or "latents" in ipadapter_model["image_proj"] or "perceiver_resampler.proj_in.weight" in ipadapter_model["image_proj"]

        easing_function = easing_functions[interpolation]

        input = [input_images1, input_images2, input_images3]
        output = []
        for input_images in input:
            if input_images == None:
                continue
            # Create pos embeds
            img_cond_embeds = clip_vision.encode_image(input_images)
            
            if is_plus:
                img_cond_embeds = img_cond_embeds.penultimate_hidden_states
            else:
                img_cond_embeds = img_cond_embeds.image_embeds
            print( f"Embed shape {img_cond_embeds.shape}")
            inbetween_embeds = []
            # Make sure we have 2 images
            if len(img_cond_embeds) > 1:
                num_embeds = len(img_cond_embeds)
                # Add beggining buffer
                inbetween_embeds.extend([img_cond_embeds[0]] * buffer)
                # Interpolate embeds
                for i in range(len(img_cond_embeds) - 1):
                    embed1 = img_cond_embeds[i]
                    embed2 = img_cond_embeds[i + 1]
                    alphas = torch.linspace(0, 1, transitioning_frames)
                    for alpha in alphas:
                        eased_alpha = easing_function(alpha.item())
                        print(f"eased alpha {eased_alpha}")
                        inbetween_embed = (1 - eased_alpha) * embed1 + eased_alpha * embed2
                        inbetween_embeds.extend([inbetween_embed] * repeat_count)
                # Add ending buffer
                inbetween_embeds.extend([img_cond_embeds[-1]] * buffer)
                # Find size of batch
                batch_size = len(inbetween_embeds)

            inbetween_embeds = torch.stack(inbetween_embeds, dim=0)
            output.append(inbetween_embeds)

        # Create empty neg embeds
        if is_plus:
            img_uncond_embeds = clip_vision.encode_image(torch.zeros([1, 224, 224, 3])).penultimate_hidden_states
        else:
            img_uncond_embeds = torch.zeros_like(img_cond_embeds)
        
        # Work out which frames to drop
        frames_to_drop = []
        if num_embeds > 2:
            for i in range(num_embeds-2):
                frames_to_drop.append(transitioning_frames*(i+1)+buffer-1)
        print(f"Frames to drop {frames_to_drop}")

        # Combine and format prompt strings
        def format_text_prompts(text_prompts):
            string = ""
            index = buffer
            for prompt in text_prompts:
                string += f"\"{index}\":\"{prompt}\",\n"
                index += transitioning_frames
            return string
        
        positive_string = format_text_prompts(positive_prompts)
        negative_string = format_text_prompts(negative_prompts)
        
        return (output[0], output[1], output[2], img_uncond_embeds, positive_string, negative_string, ipadapter, batch_size, frames_to_drop,)

```
