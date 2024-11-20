# Google Translate CLIP Text Encode Node
## Documentation
- Class name: `GoogleTranslateCLIPTextEncodeNode`
- Category: `AlekPet Nodes/conditioning`
- Output node: `False`

This node leverages the Google Translate API to translate text from one language to another and then encodes the translated text using the CLIP model for further processing or analysis. It supports automatic language detection, manual translation control, and integrates CLIP's powerful text and image understanding capabilities to produce conditioning vectors and pooled outputs for downstream tasks.
## Input types
### Required
- **`from_translate`**
    - Specifies the source language for translation or 'auto' for automatic language detection. It plays a crucial role in guiding the translation process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `Union[str, List[str]]`
- **`to_translate`**
    - Defines the target language for the translation, with 'en' (English) as the default. This parameter determines the language into which the text will be translated.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`manual_translate`**
    - A boolean flag that, when set to True, bypasses the translation process and uses the original text for CLIP encoding. This allows for optional use of the translation feature.
    - Comfy dtype: `COMBO[BOOLEAN]`
    - Python dtype: `bool`
- **`text`**
    - The text to be translated and/or encoded. This is the primary input for the translation and encoding process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - A CLIP model instance used for encoding the translated text into vectors. This enables the integration of text understanding and image recognition capabilities.
    - Comfy dtype: `CLIP`
    - Python dtype: `CLIP`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - A vector representation of the translated text, suitable for conditioning models or further analysis.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]`
- **`string`**
    - Comfy dtype: `STRING`
    - The translated text, providing the outcome of the translation process.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class GoogleTranslateCLIPTextEncodeNode:

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "from_translate": (
                    ["auto"] + list(LANGUAGES.keys()),
                    {"default": "auto"},
                ),
                "to_translate": (list(LANGUAGES.keys()), {"default": "en"}),
                "manual_translate": ([True, False],),
                "text": ("STRING", {"multiline": True, "placeholder": "Input prompt"}),
                "clip": ("CLIP",),
            }
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "STRING",
    )
    FUNCTION = "translate_text"
    CATEGORY = "AlekPet Nodes/conditioning"

    def translate_text(self, **kwargs):
        from_translate = kwargs.get("from_translate")
        to_translate = kwargs.get("to_translate")
        manual_translate = kwargs.get("manual_translate", False)
        text = kwargs.get("text")
        clip = kwargs.get("clip")

        text_tranlsated = (
            translate(text, from_translate, to_translate)
            if not manual_translate
            else text
        )
        tokens = clip.tokenize(text_tranlsated)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], text_tranlsated)

```
