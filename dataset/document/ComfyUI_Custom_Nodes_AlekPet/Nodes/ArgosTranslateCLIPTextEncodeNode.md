# Argos Translate CLIP Text Encode Node
## Documentation
- Class name: `ArgosTranslateCLIPTextEncodeNode`
- Category: `AlekPet Nodes/conditioning`
- Output node: `False`

This node is designed for translating text from one language to another and encoding the translated text using the CLIP model for further processing or conditioning in AI models. It supports a variety of languages and leverages the CLIP model to provide a rich, contextual representation of the translated text.
## Input types
### Required
- **`from_translate`**
    - Specifies the source language from which the text will be translated. It determines the starting point of the translation process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`to_translate`**
    - Defines the target language to which the text will be translated. It affects the final output of the translation process by setting the desired language.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`text`**
    - The text to be translated. This input is crucial as it provides the content that will undergo translation and subsequent encoding.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - A CLIP model instance used for encoding the translated text. It plays a key role in generating contextual embeddings for the text.
    - Comfy dtype: `CLIP`
    - Python dtype: `CLIP`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - Provides the conditioning information derived from the CLIP model's encoding of the translated text.
    - Python dtype: `List[List[torch.Tensor, Dict[str, torch.Tensor]]]`
- **`string`**
    - Comfy dtype: `STRING`
    - The translated text, serving as a direct output of the translation process.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ArgosTranslateCLIPTextEncodeNode:
    @classmethod
    def INPUT_TYPES(self):
        self.langs_support = ALL_CODES["russian"]["targets"]
        return {
            "required": {
                "from_translate": (list(ALL_CODES.keys()), {"default": "russian"}),
                "to_translate": (self.langs_support, {"default": "english"}),
                "text": ("STRING", {"multiline": True, "placeholder": "Input text"}),
                "clip": ("CLIP",),
            }
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "STRING",
    )
    FUNCTION = "argos_translate_text"
    CATEGORY = "AlekPet Nodes/conditioning"

    def argos_translate_text(self, from_translate, to_translate, text, clip):
        self.langs_support = ALL_CODES[from_translate]["targets"]
        text = translate(text, from_translate, to_translate)
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], text)

    @classmethod
    def VALIDATE_INPUTS(cls, from_translate, to_translate, text, clip):
        return True

```
