# Prompt
## Documentation
- Class name: `easy prompt`
- Category: `EasyUse/Prompt`
- Output node: `False`

The `easy prompt` node is designed to facilitate the generation of text prompts by allowing users to input multiple strings and optionally combine them into a structured prompt. This node simplifies the process of creating complex or structured text inputs for further processing or generation tasks.
## Input types
### Required
- **`prompt`**
    - Accepts a string input to be used as part of a prompt, providing flexibility in generating text-based inputs.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`main`**
    - Serves as the primary input for prompt generation, enabling the combination of multiple text elements.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`lighting`**
    - Takes a string input to incorporate lighting conditions into the prompt, further expanding the possibilities for prompt customization.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`prompt`**
    - Comfy dtype: `STRING`
    - Outputs a structured prompt, aggregating the input strings into a cohesive format.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class prompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Prompt"}),
            "main": ([
                'none',
                'beautiful woman, detailed face',
                'handsome man, detailed face',
                'pretty girl',
                'handsome boy',
                'dog',
                'cat',
                'Buddha',
                'toy'
            ], {"default": "none"}),
            "lighting": ([
                'none',
                'sunshine from window',
                'neon light, city',
                'sunset over sea',
                'golden time',
                'sci-fi RGB glowing, cyberpunk',
                'natural lighting',
                'warm atmosphere, at home, bedroom',
                'magic lit',
                'evil, gothic, Yharnam',
                'light and shadow',
                'shadow from window',
                'soft studio lighting',
                'home atmosphere, cozy bedroom illumination',
                'neon, Wong Kar-wai, warm',
                'cinemative lighting',
                'neo punk lighting, cyberpunk',
            ],{"default":'none'})
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Prompt"

    def doit(self, prompt, main, lighting):
        if has_chinese(prompt):
            prompt = zh_to_en([prompt])[0]
        if lighting != 'none' and main != 'none':
            prompt = main + ',' + lighting + ',' + prompt
        elif lighting != 'none' and main == 'none':
            prompt = prompt + ',' + lighting
        elif main != 'none':
            prompt = main + ',' + prompt

        return prompt,

```
