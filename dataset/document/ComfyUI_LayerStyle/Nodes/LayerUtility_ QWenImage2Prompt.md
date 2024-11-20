# LayerUtility: QWenImage2Prompt
## Documentation
- Class name: `LayerUtility: QWenImage2Prompt`
- Category: `😺dzNodes/LayerUtility/Prompt`
- Output node: `False`

This node integrates a chat model to generate text prompts based on an input image and a question. It utilizes an image-to-text conversion process to facilitate interactions with the chat model, aiming to produce descriptive or query-responsive text outputs.
## Input types
### Required
- **`image`**
    - The input image for which the text prompt will be generated. It serves as the visual context for the chat model to interpret and describe.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`question`**
    - An optional question to guide the generation of the text prompt, allowing for more specific or directed outputs from the chat model.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`text`**
    - Comfy dtype: `STRING`
    - The generated text prompt that describes or answers the question about the input image.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class QWenImage2Prompt:
    def __init__(self):
        self.chat_model = UformGen2QwenChat()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "question": ("STRING", {"multiline": False, "default": "describe this image",},),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "uform_gen2_qwen_chat"
    CATEGORY = '😺dzNodes/LayerUtility/Prompt'

    def uform_gen2_qwen_chat(self, image, question):
        history = []  # Example empty history
        pil_image = ToPILImage()(image[0].permute(2, 0, 1))
        temp_path = files_for_uform_gen2_qwen / "temp.png"
        pil_image.save(temp_path)
        
        response = self.chat_model.chat_response(question, history, temp_path)
        return (response.split("assistant\n", 1)[1], )

```
