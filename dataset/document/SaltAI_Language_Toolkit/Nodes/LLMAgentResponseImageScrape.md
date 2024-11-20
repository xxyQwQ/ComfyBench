---
tags:
- LLM
---

# ∞ Agent Response Image Scrape
## Documentation
- Class name: `LLMAgentResponseImageScrape`
- Category: `SALT/Language Toolkit/Agents/Tools`
- Output node: `False`

This node is designed to extract and process images from a given text response, identifying and converting any embedded image paths into a tensor representation suitable for further image processing or analysis.
## Input types
### Required
- **`response_string`**
    - The response string is the primary input for this node, containing the text from which images will be scraped. It plays a crucial role in determining which images are extracted and processed, directly affecting the node's output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a tensor representation of all the images found and processed from the input text. This tensor can be used for further image processing or analysis tasks.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMAgentResponseImageScrape:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "response_string": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "scrape_response"

    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Agents/Tools"

    def scrape_response(self, response_string):
        pattern = r'<img\s+([^>]+)>'
        matches = re.findall(pattern, response_string)
        added_paths = []

        images = []
        for match in matches:
            logger.info(f"Found match {match}")
            path = match.strip()
            if os.path.exists(path) and path not in added_paths:
                logger.info("Adding to image pool.")
                img = Image.open(path)
                images.append(img)
                added_paths.append(path)

        images = [pil2tensor(img) for img in images]
        if images:
            images = torch.cat(images, dim=0)
        else:
            images = pil2tensor(Image.new("RGB", (512, 512), (0, 0, 0)))

        return (images,)

```
