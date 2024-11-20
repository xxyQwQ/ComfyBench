# GenerateCategoryName
## Documentation
- Class name: `GenerateCategoryName`
- Category: `SALT/CLUSTERING`
- Output node: `False`

This node is designed to generate a marketable and specific name for a product category based on provided product descriptions and an existing product category tree. It analyzes common themes, attributes, or characteristics from the descriptions to ideate and select a category name that is clear, specific, and appealing to customers.
## Input types
### Required
- **`captions`**
    - Captions are textual descriptions of products that serve as the basis for identifying common themes and characteristics to generate a category name. They play a crucial role in the ideation process by providing the content for analysis.
    - Comfy dtype: `LIST`
    - Python dtype: `List[str]`
- **`llm`**
    - The language model used to assist in generating category names by analyzing the provided captions and existing category tree for insights and suggestions.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `Dict`
- **`existing_tree`**
    - The current structure of product categories, which is considered during the generation process to ensure the new category name fits within the existing hierarchy and does not duplicate or conflict with current categories.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`category_name`**
    - Comfy dtype: `STRING`
    - The generated name for the product category, selected for its clarity, specificity, and marketability based on the analysis of product descriptions and the existing category tree.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class GenerateCategoryName:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "captions": ("LIST", ),
                "llm": ("LLM_MODEL", ),
                "existing_tree": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("category_name",)

    FUNCTION = "generate_name"
    CATEGORY = f"{MENU_NAME}/CLUSTERING"

    def generate_name(self, captions, llm, existing_tree):
        try:
            captions = np.random.choice(captions, min(300, len(captions)))
            captions_str = ";\n".join(captions)    
            prompt = """
    Here descriptions of goods from a single marketplace category:

    <descriptions>""" + captions_str + """
    </descriptions>

    Here is an existing tree, pay attention on that, don't generate duplicates:
    <existing_tree>""" + existing_tree + """
    </existing_tree>

    Your task is to generate a clear, specific and marketable name for the category that these goods belong to. 

    First, carefully analyze the provided descriptions to identify common themes, attributes, or characteristics that could define the category. Look for patterns in the types of products, their uses, materials, or target audiences.

    Next, brainstorm a list of potential category names based on the themes and characteristics you identified. Write out your brainstormed names inside <brainstorm> tags. Focus on names that are descriptive, specific and engaging.

    Now, select the best category name from your brainstormed options. Consider factors like:
    - Clarity: Will customers immediately understand what types of goods are in this category? 
    - Specificity: Is the name specific enough to differentiate this category from others, without being so narrow that it excludes relevant goods?
    - Marketability: Is the name engaging and appealing? Would it draw customers to click on and browse this category?

    Provide your selected category name inside <category> tags.

    Provide the primary information about this category. For instance, if the category includes travel and fashion bags, simply state "Bags." If it consists of modern and minimalist tables, state "Tables." Emphasize the main object of this category!
        """
            answer = llm['llm'].complete(prompt).text.replace("\n", "")
            category_name = extract_text_between_tags(answer)[0]
        except:
            return ("", )
        return (category_name,)

```
