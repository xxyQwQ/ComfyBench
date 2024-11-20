---
tags:
- Prompt
- PromptStyling
---

# SD3 Text Conditioning With Options. One Prompt (Mikey)
## Documentation
- Class name: `SD3TextConditioningWithOptionsOnePrompt`
- Category: `Mikey/SD3/Conditioning`
- Output node: `False`

The SD3TextConditioningWithOptionsOnePrompt node is designed to enhance text prompts for image generation by applying style-based modifications and conditioning. It leverages specific styles to modify positive and negative prompts, integrates seed-based variations for style selection, and utilizes advanced text encoding techniques to refine the conditioning process for improved image generation outcomes.
## Input types
### Required
- **`positive_prompt`**
    - Represents the base positive prompt that will be modified according to the selected style and further processed to enhance its effectiveness in guiding the image generation process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`negative_prompt`**
    - Serves as the base negative prompt that undergoes style-based modifications and processing to refine its role in the image generation process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - Specifies the CLIP model to be used for encoding the prompts, playing a crucial role in the conditioning process.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.nn.Module`
- **`option_positive_clip_l`**
    - Option to apply specific CLIP model settings for the positive prompt's local features.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `dict`
- **`option_positive_clip_g`**
    - Option to apply specific CLIP model settings for the positive prompt's global features.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `dict`
- **`option_positive_t5xxl`**
    - Option to apply specific T5-XXL model settings for the positive prompt.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `dict`
- **`option_negative_clip_l`**
    - Option to apply specific CLIP model settings for the negative prompt's local features.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `dict`
- **`option_negative_clip_g`**
    - Option to apply specific CLIP model settings for the negative prompt's global features.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `dict`
- **`option_negative_t5xxl`**
    - Option to apply specific T5-XXL model settings for the negative prompt.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `dict`
- **`padding_character`**
    - Defines the character used for padding in the text encoding process, affecting the alignment and spacing within the encoded prompts.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`positive_conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The modified positive prompt, enhanced with style-based modifications and advanced encoding techniques for improved image generation.
    - Python dtype: `str`
- **`negative_conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The enhanced negative prompt, similarly modified with style-based adjustments and encoding to refine the image generation process.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SD3TextConditioningWithOptionsOnePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'positive_prompt': ('STRING', {'default': 'positive prompt', 'multiline': True}),
                             'negative_prompt': ('STRING', {'default': 'negative prompt', 'multiline': True}),
                             'clip': ('CLIP',),
                             'option_positive_clip_l': (['Unmodified','Padded','Empty'], {'default': 'Unmodified'}),
                             'option_positive_clip_g': (['Unmodified','Padded','Empty'], {'default': 'Unmodified'}),
                             'option_positive_t5xxl': (['Unmodified','Padded','Empty'], {'default': 'Unmodified'}),
                             'option_negative_clip_l': (['Unmodified','Padded','Empty'], {'default': 'Unmodified'}),
                             'option_negative_clip_g': (['Unmodified','Padded','Empty'], {'default': 'Unmodified'}),
                             'option_negative_t5xxl': (['Unmodified','Padded','Empty'], {'default': 'Unmodified'}),
                             'padding_character': ('STRING', {'default': ','})}}

    RETURN_TYPES = ('CONDITIONING','CONDITIONING',)
    RETURN_NAMES = ('positive_conditioning','negative_conditioning')
    FUNCTION = 'process'
    CATEGORY = 'Mikey/SD3/Conditioning'

    def process(self, positive_prompt, negative_prompt, clip, option_positive_clip_l, option_positive_clip_g, option_positive_t5xxl, option_negative_clip_l, option_negative_clip_g, option_negative_t5xxl, padding_character):
        positive_tokens = clip.tokenize(positive_prompt)
        if len(positive_prompt) == 0 or option_positive_clip_g == 'Empty':
            positive_tokens['g'] = []
        if len(positive_tokens['g']) < 77 and option_positive_clip_l == 'Padded':
            padded_g_string = positive_prompt + padding_character * (77 - len(positive_tokens['g']))
            positive_tokens['g'] = clip.tokenize(padded_g_string)['g']

        if len(positive_prompt) == 0 or option_positive_clip_l == 'Empty':
            positive_tokens['l'] = []
        elif len(positive_tokens['l']) < 77 and option_positive_clip_l == 'Padded':
            padded_l_string = positive_prompt + padding_character * (77 - len(positive_tokens['l']))
            positive_tokens['l'] = clip.tokenize(padded_l_string)['l']
        else:
            positive_tokens['l'] = clip.tokenize(positive_prompt)['l']

        if len(positive_prompt) == 0 or option_positive_t5xxl == 'Empty':
            positive_tokens['t5xxl'] = []
        elif len(positive_tokens['t5xxl']) < 512 and option_positive_t5xxl == 'Padded':
            padded_t5xxl_string = positive_prompt + padding_character * (512 - len(positive_tokens['t5xxl']))
            positive_tokens['t5xxl'] = clip.tokenize(padded_t5xxl_string)['t5xxl']
        else:
            positive_tokens['t5xxl'] = clip.tokenize(positive_prompt)['t5xxl']

        if len(positive_tokens['l']) != len(positive_tokens['g']):
            empty = clip.tokenize('')
            while len(positive_tokens['l']) < len(positive_tokens['g']):
                positive_tokens['l'] += empty['l']
            while len(positive_tokens['l']) > len(positive_tokens['g']):
                positive_tokens['g'] += empty['g']
        positive_conditioning, positive_pooled = clip.encode_from_tokens(positive_tokens, return_pooled=True)

        # now negative
        negative_tokens = clip.tokenize(negative_prompt)
        if len(negative_prompt) == 0 or option_negative_clip_g == 'Empty':
            negative_tokens['g'] = []
        if len(negative_tokens['g']) < 77 and option_negative_clip_l == 'Padded':
            padded_g_string = negative_prompt + padding_character * (77 - len(negative_tokens['g']))
            negative_tokens['g'] = clip.tokenize(padded_g_string)['g']

        if len(negative_prompt) == 0 or option_negative_clip_l == 'Empty':
            negative_tokens['l'] = []
        elif len(negative_tokens['l']) < 77 and option_negative_clip_l == 'Padded':
            padded_l_string = negative_prompt + padding_character * (75 - len(negative_tokens['l']))
            negative_tokens['l'] = clip.tokenize(padded_l_string)['l']
        else:
            negative_tokens['l'] = clip.tokenize(negative_prompt)['l']

        if len(negative_prompt) == 0 or option_negative_t5xxl == 'Empty':
            negative_tokens['t5xxl'] = []
        elif len(negative_tokens['t5xxl']) < 512 and option_negative_t5xxl == 'Padded':
            padded_t5xxl_string = negative_prompt + padding_character * (512 - len(negative_tokens['t5xxl']))
            negative_tokens['t5xxl'] = clip.tokenize(padded_t5xxl_string)['t5xxl']
        else:
            negative_tokens['t5xxl'] = clip.tokenize(negative_prompt)['t5xxl']

        if len(negative_tokens['l']) != len(negative_tokens['g']):
            empty = clip.tokenize('')
            while len(negative_tokens['l']) < len(negative_tokens['g']):
                negative_tokens['l'] += empty['l']
            while len(negative_tokens['l']) > len(negative_tokens['g']):
                negative_tokens['g'] += empty['g']
        negative_conditioning, negative_pooled = clip.encode_from_tokens(negative_tokens, return_pooled=True)

        return (
            [[positive_conditioning, {'pooled_output': positive_pooled}]],
            [[negative_conditioning, {'pooled_output': negative_pooled}]],
        )

```
