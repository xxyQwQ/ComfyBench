# Create Taxonomy
## Documentation
- Class name: `CreateTaxonomy`
- Category: `SALT/TAXONOMY`
- Output node: `False`

The CreateTaxonomy node is designed to automate the process of updating and generating a new taxonomy based on provided product descriptions and updates. It leverages a language model to suggest modifications to an existing taxonomy or create a new one, aiming to enhance the organization and categorization of products within a marketplace.
## Input types
### Required
- **`path_to_descriptions`**
    - Specifies the file path to the product descriptions, which are used as input for generating or updating the taxonomy.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`updates`**
    - Indicates the number of updates to be applied to the taxonomy during the generation process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`batch_size`**
    - Determines the number of product descriptions processed in each batch for taxonomy generation or update.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`update_rate`**
    - Specifies the frequency at which the taxonomy is updated based on the provided product descriptions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`fix_request`**
    - A template for the request to fix issues in the taxonomy, used by the language model to suggest specific changes.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`prompt`**
    - A template for the prompt used by the language model to categorize products and suggest taxonomy updates.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`llm`**
    - The language model used for generating taxonomy suggestions and updates.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `object`
### Optional
- **`path_to_taxonomy`**
    - Optional. Specifies the file path to the existing taxonomy, which can be updated or used as a reference for generating a new taxonomy.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`taxonomy`**
    - Comfy dtype: `STRING`
    - The updated or newly generated taxonomy as a result of the node's execution.
    - Python dtype: `str`
- **`old_taxonomy`**
    - Comfy dtype: `STRING`
    - The original taxonomy before any updates were applied, used for comparison or rollback purposes.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CreateTaxonomy:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path_to_descriptions": ("STRING", {"default": "", "multiline": False}),
                "updates": ("INT", {"default": 10, "min": -1}),
                "batch_size": ("INT", {"default": 20, "min": 1}),
                "update_rate": ("INT", {"default": 10, "min": 1}),
                "fix_request": ("STRING", {"default": """
I am working on improving the taxonomy for the Takealot.com marketplace and need some assistance.

Below is the current taxonomy:
{}

Here is the feedback received:
{}

Please help me rewrite the taxonomy. Share your thoughts on the improvements first, and then provide the commands to enhance or fix this taxonomy within the <commands></commands> tags. Start each command on a new line.
There are three types of commands: delete category, add category, move one category to another, and rename category.
delete will delete the selected category, and add will create the provided folder.

Use these commands as follows. Here is an example:
<commands>
delete: Path>To>Category>To>Delete
delete: Path>To>Category>To>Delete
add: Path>To>Category>To>Add
add: Path>To>Category>To>Add
add: Path>To>Category>To>Add
move: Path>To>Category>To>Add $ Path>To>Category>To>Add
move: Path>To>Category>To>Add $ Path>To>Category>To>Add
rename: Path>To>Category>To>Add $ Path>To>Category>To>Add
rename: Path>To>Category>To>Add $ Path>To>Category>To>Add
</commands>

Use $ as a separate between two categories for move and rename commands.
Use commands! Don't provide JSON. 

Note that some of the feedback may contain noise or less valuable insights. Focus on fixing errors that are repeated several times.

I provided a lot of feedback. Please select only the most valuable points that are mentioned multiple times and write no more than 20 commands! No more than 20 commands! I already have a very good taxonomy, so please don't change it dramatically! I just want to refine it a little bit. If you want to delete something or add something, provide your explanation!

This is very important to me, so be very careful when changing this taxonomy. I'm very afraid of doing something wrong; my life depends on it! You provided only a chunk of feedback about several products. I used 9 million items to create this taxonomy, so remember that you can change it a little. But I really want to change it, to improve it, to make it better!

Edit the taxonomy only based on the provided feedback!

Please avoid to create categories with name "Other" or "Miscellaneous". Be precise in naming categories, but don't create too many of them.
                """, "multiline": True}),
                "prompt": ("STRING", {"default": """
I am working on improving the taxonomy for the Takealot.com marketplace and need some assistance.
Here is a description of the products:
{}
Please try to categorize these products using the provided taxonomy. Below is my current taxonomy:
{}
If there isn't a suitable category for an item, feel free to suggest a new category.
Please provide category in which you would like to place this item within the <category> </category> tags, split folders with '>'.
Please provide your final feedback about taxonomy within the <feedback> </feedback> tags.
Include suggestions on how to improve this taxonomy based on the provided items, but do not make drastic changes.
Do not provide a new taxonomy. Provide feedback only, and I will create the new taxonomy from that.
                """, "multiline": True}),
                "llm": ("LLM_MODEL", ),
            },
            "optional": {
                "path_to_taxonomy": ("STRING", {"default": "", "multiline": False}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("taxonomy", "old_taxonomy")

    FUNCTION = "generate_taxonomy"
    CATEGORY = f"{MENU_NAME}/TAXONOMY"

    def generate_taxonomy(self, path_to_descriptions, updates, batch_size, update_rate, prompt, fix_request, llm, path_to_taxonomy=None):
        old_timeout = llm['llm'].timeout
        llm['llm'].timeout = 500
        if path_to_taxonomy is not None and path_to_taxonomy != "":
            with open(path_to_taxonomy) as json_file:
                current_taxonomy = json.dumps(json.load(json_file))
        else:
            current_taxonomy = "{}"
        old_taxonomy = copy.deepcopy(current_taxonomy)

        with open(path_to_descriptions) as json_file:
            d = json.load(json_file)
        data = []
        for k, v in d.items():
            if "captions" in v:
                data.append({"image": k, "captions": json.loads(v)['captions']})
                
        feedbacks = []
        global_idx = 0
        global_updates = 0
        for batch in range(len(data) // batch_size):
            if global_updates >= updates:
                break
            tries = 0
            while tries < 10:
                tries += 1
                try:
                    captions = []
                    for item in range(batch_size):
                        captions.append(data[global_idx % len(data)]['captions'][0])
                        global_idx += 1
                    feedback = llm["llm"].complete(prompt.format("\n---------------------------\n".join(captions), current_taxonomy)).text
                    feedback = extract_text_between_tags(feedback, start_tag="<feedback>", end_tag="</feedback>")[0]
                    feedbacks.append(feedback)
                    break
                except Exception as e:
                    print("An error occurred:", e)
                    continue
            
            if (batch + 1) % update_rate == 0:
                tries = 0
                while tries < 10:
                    tries += 1
                    try:
                        commands = llm["llm"].complete(fix_request.format(current_taxonomy, "\n".join(feedbacks))).text
                        print(commands)
                        if "</commands>" not in commands:
                            commands += "</commands>"
                        commands = extract_text_between_tags(commands, start_tag="<commands>", end_tag="</commands>")[0]
                        taxonomy_json = json.loads(current_taxonomy)
                        command_list = set(commands.strip().split('\n'))
                        for command in command_list:
                            action, path = command.split(':', 1)
                            path = path.strip()
                            if action.strip().lower() == 'delete':
                                delete_category(path, taxonomy_json)
                            elif action.strip().lower() == 'add':
                                add_category(path, taxonomy_json)
                            elif action.strip().lower() == 'move':
                                source_path, dest_path = path.split('$')
                                move_category(source_path.strip(), dest_path.strip(), taxonomy_json)
                        
                        current_taxonomy = json.dumps(taxonomy_json, indent=4)
                        feedbacks = []
                        global_updates += 1
                        break
                    except Exception as e:
                        print("An error occurred:", e)
                        continue
            print("global_updates", global_updates)
        
        llm['llm'].timeout = old_timeout
        return (current_taxonomy, old_taxonomy)

```
