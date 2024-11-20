# LayerUtility: ImageRewardFilter
## Documentation
- Class name: `LayerUtility: ImageRewardFilter`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

The ImageRewardFilter node filters and ranks images based on their relevance to a given prompt using a reward model. It selects the top images as per the specified output number, effectively curating a subset of images that best match the prompt criteria.
## Input types
### Required
- **`images`**
    - A collection of images to be evaluated and filtered based on their relevance to the prompt. This input is crucial for determining which images are retained and which are discarded, based on their scores from the reward model.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`prompt`**
    - The text prompt against which the images are evaluated. This prompt guides the reward model in scoring the images, influencing which images are considered most relevant.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`output_num`**
    - Specifies the number of top-scoring images to retain. This parameter controls the size of the output image set, ensuring only the most relevant images are selected.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`images`**
    - Comfy dtype: `IMAGE`
    - The subset of input images that were ranked highest by the reward model, deemed most relevant to the prompt.
    - Python dtype: `List[torch.Tensor]`
- **`obsolete_images`**
    - Comfy dtype: `IMAGE`
    - Images that were scored lower by the reward model and not included in the top selections based on the output number.
    - Python dtype: `List[torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageRewardFilter:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "images": ("IMAGE", ),
                "prompt": ("STRING", {"multiline": False}),
                "output_num": ("INT", {"default": 3, "min": 1, "max": 999999, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("images", 'obsolete_images',)
    FUNCTION = 'image_reward_filter'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_reward_filter(self, images, prompt, output_num,):
        log(f"len(images)= {len(images)}, output_num={output_num}")
        if output_num > len(images):
            log(f"Error: {NODE_NAME} skipped, because 'output_num' is greater then input images.", message_type='error')
            return (images,)

        scores = []
        ret_images = []
        obsolete_images = []

        if not torch.cuda.is_available() :
            device = "cpu"
        else:
            device = "cuda"

        import ImageReward as RM
        reward_model = RM.load("ImageReward-v1.0")
        reward_model = reward_model.to(device=device)

        with torch.no_grad():
            for i in range(len(images)):
                score = reward_model.score(prompt, tensor2pil(images[i]))
                scores.append(
                    {
                        "score":score,
                        "image_index":i
                    }
                )
        scores = sorted(scores, key=lambda s: s['score'], reverse=True)

        for i in range(len(images)):
            if i < output_num:
                log(f"{NODE_NAME} append image #{i}: {scores[i]['image_index']}, score = {scores[i]['score']}.")
                ret_images.append(images[scores[i]['image_index']])
            else:
                log(f"{NODE_NAME} obsolete image #{i}: {scores[i]['image_index']}, score = {scores[i]['score']}.")
                obsolete_images.append(images[scores[i]['image_index']])

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')

        return (ret_images, obsolete_images,)

```
