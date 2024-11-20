# Wildcard Prompt from String
## Documentation
- Class name: `WildcardPromptFromString`
- Category: `ImpactPack/Util`
- Output node: `False`

This node is designed to generate and process text prompts with wildcards, enabling dynamic content creation by replacing specified wildcard patterns with appropriate values. It facilitates the customization and variation of text inputs for further processing or output generation, emphasizing flexibility and adaptability in text manipulation.
## Input types
### Required
- **`string`**
    - The primary text input that may contain wildcards or placeholders for dynamic content generation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`delimiter`**
    - Specifies the delimiter used to identify or separate wildcards within the text, enabling precise pattern recognition and processing.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`prefix_all`**
    - A prefix applied to all wildcards before processing, aiding in their identification and manipulation within the text.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`postfix_all`**
    - A postfix applied to all wildcards after processing, ensuring that replaced content is correctly formatted or distinguished.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`restrict_to_tags`**
    - Limits wildcard processing to only those tags specified, allowing for targeted content generation and customization.
    - Comfy dtype: `STRING`
    - Python dtype: `list`
- **`exclude_tags`**
    - Excludes specific tags from wildcard processing, providing control over which content remains static.
    - Comfy dtype: `STRING`
    - Python dtype: `list`
## Output types
- **`wildcard`**
    - Comfy dtype: `STRING`
    - The processed text with wildcards replaced by their corresponding values, ready for use in various applications.
    - Python dtype: `str`
- **`segs_labels`**
    - Comfy dtype: `STRING`
    - Labels or segments generated during the wildcard processing, offering insights into the structure and content of the processed text.
    - Python dtype: `list`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class WildcardPromptFromString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": ("STRING", {"forceInput": True}),
                "delimiter": ("STRING", {"multiline": False, "default": "\\n" }),
                "prefix_all": ("STRING", {"multiline": False}),
                "postfix_all": ("STRING", {"multiline": False}),
                "restrict_to_tags": ("STRING", {"multiline": False}),
                "exclude_tags": ("STRING", {"multiline": False})
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("wildcard", "segs_labels",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, string, delimiter, prefix_all, postfix_all, restrict_to_tags, exclude_tags):
        # convert \\n to newline character
        if delimiter == "\\n":
            delimiter = "\n"

        # some sanity checks and normalization for later processing
        if prefix_all is None:
            prefix_all = ""
        if postfix_all is None:
            postfix_all = ""
        if restrict_to_tags is None:
            restrict_to_tags = ""
        if exclude_tags is None:
            exclude_tags = ""

        restrict_to_tags = restrict_to_tags.split(", ")
        exclude_tags = exclude_tags.split(", ")

        # build the wildcard prompt per list entry
        output = ["[LAB]"]
        labels = []
        for x in string.split(delimiter):
            label = str(len(labels) + 1)
            labels.append(label)
            x = x.split(", ")
            # restrict to tags
            if restrict_to_tags != [""]:
                x = list(set(x) & set(restrict_to_tags))
            # remove tags
            if exclude_tags != [""]:
                x = list(set(x) - set(exclude_tags))
            # next row: <LABEL> <PREFIX> <TAGS> <POSTFIX>
            prompt_for_seg = f'[{label}] {prefix_all} {", ".join(x)} {postfix_all}'.strip()
            output.append(prompt_for_seg)
        output = "\n".join(output)

        # clean string: fixup double spaces, commas etc.
        output = re.sub(r' ,', ',', output)
        output = re.sub(r'  +', ' ', output)
        output = re.sub(r',,+', ',', output)
        output = re.sub(r'\n, ', '\n', output)

        return output, ", ".join(labels)

```
