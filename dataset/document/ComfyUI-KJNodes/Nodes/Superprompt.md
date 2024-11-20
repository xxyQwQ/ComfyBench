---
tags:
- Prompt
---

# Superprompt
## Documentation
- Class name: `Superprompt`
- Category: `KJNodes/text`
- Output node: `False`

Superprompt is designed to enhance text prompts by upsampling them into more detailed descriptions using a T5 model fine-tuned on the SuperPrompt dataset. It serves as a pre-generation step to enrich prompts for text-to-image models, facilitating the creation of more detailed and accurate images.
## Input types
### Required
- **`instruction_prompt`**
    - A guiding instruction to shape the expansion of the prompt, setting the context for how the text should be detailed.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`prompt`**
    - The initial text prompt to be expanded into a more detailed description.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`max_new_tokens`**
    - Specifies the maximum number of new tokens to be generated, controlling the length of the expanded description.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The enhanced, detailed version of the original text prompt.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class Superprompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "instruction_prompt": ("STRING", {"default": 'Expand the following prompt to add more detail', "multiline": True}),
                "prompt": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
                "max_new_tokens": ("INT", {"default": 128, "min": 1, "max": 4096, "step": 1}),
            } 
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "KJNodes/text"
    DESCRIPTION = """
# SuperPrompt
A T5 model fine-tuned on the SuperPrompt dataset for  
upsampling text prompts to more detailed descriptions.  
Meant to be used as a pre-generation step for text-to-image  
models that benefit from more detailed prompts.  
https://huggingface.co/roborovski/superprompt-v1
"""

    def process(self, instruction_prompt, prompt, max_new_tokens):
        device = model_management.get_torch_device()
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        checkpoint_path = os.path.join(script_directory, "models","superprompt-v1")
        if not os.path.exists(checkpoint_path):
                print(f"Downloading model to: {checkpoint_path}")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="roborovski/superprompt-v1", 
                                  local_dir=checkpoint_path, 
                                  local_dir_use_symlinks=False)
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small", legacy=False)

        model = T5ForConditionalGeneration.from_pretrained(checkpoint_path, device_map=device)
        model.to(device)
        input_text = instruction_prompt + ": " + prompt
  
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(input_ids,  max_new_tokens=max_new_tokens)
        out = (tokenizer.decode(outputs[0]))
        out = out.replace('<pad>', '')
        out = out.replace('</s>', '')
        
        return (out, )

```
