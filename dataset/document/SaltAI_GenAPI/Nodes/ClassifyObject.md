# ClassifyObject
## Documentation
- Class name: `ClassifyObject`
- Category: `SALT/CLUSTERING`
- Output node: `False`

The ClassifyObject node is designed to classify images into categories based on a given taxonomy and a multimodal model. It leverages an image URL and a taxonomy description to generate a category name for the image, utilizing a large language model capable of understanding both text and image inputs.
## Input types
### Required
- **`image_url`**
    - The URL of the image to be classified. It serves as a direct link to the image that the model will analyze to determine its category.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`taxonomy`**
    - A string representing the taxonomy against which the image will be classified. This taxonomy guides the model in understanding the possible categories and their hierarchy.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`llm`**
    - A configuration object for the large language model (LLM) used in the classification. It includes the model details and API key necessary for multimodal understanding.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `dict`
## Output types
- **`category_name`**
    - Comfy dtype: `STRING`
    - The name of the category into which the image has been classified, based on the provided taxonomy and analyzed by the multimodal model.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ClassifyObject:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_url": ("STRING", {"default": "link to image"}),
                "taxonomy": ("STRING", {"default": "Taxonomy", "multiline": True}),
                "llm": ("LLM_MODEL", ),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("category_name",)

    FUNCTION = "generate_name"
    CATEGORY = f"{MENU_NAME}/CLUSTERING"

    def generate_name(self, image_url, taxonomy, llm):
        image_documents = load_image_urls([image_url])
        openai_mm_llm = OpenAIMultiModal(
            model=llm["llm"].model, api_key=llm["llm"].api_key, max_new_tokens=300
        )
        complete_response = openai_mm_llm.complete(
            prompt=f"""Here is a marketplace taxonomy: {taxonomy}
            In which category would you like to place item on the image? Provide full path. And only path, without explanation.
            Split folders in a path with ' > '. 
            """,
            image_documents=image_documents,
        )        
        return (complete_response.text,)

```
