---
tags:
- LLM
---

# ∞ Simple ComfyUI Node Drafter
## Documentation
- Class name: `LLMCustomNodeComposer`
- Category: `SALT/Language Toolkit/Tools`
- Output node: `False`

This node is designed to draft simple ComfyUI nodes by generating the necessary code structure based on provided specifications. It abstracts the complexities of node creation, allowing users to focus on defining the functionality and inputs of their custom nodes.
## Input types
### Required
- **`llm_model`**
    - Specifies the language model to be used for generating the code structure. It is crucial for interpreting the input specifications and composing the appropriate code.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `Dict[str, Any]`
- **`class_prefix`**
    - Defines the prefix for the class name of the generated node, allowing for customization and easy identification.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`node_explanation`**
    - A brief description or explanation of the node's intended functionality, providing context for the code generation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`example_documents`**
    - An optional list of example documents that can be used as references or templates in the node drafting process.
    - Comfy dtype: `DOCUMENT`
    - Python dtype: `List[Document]`
- **`example_code`**
    - Optional example code that can serve as a guideline or inspiration for the generated node's functionality.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`response`**
    - Comfy dtype: `STRING`
    - The output is the generated code for the ComfyUI node, encapsulating the structure and logic as specified by the input parameters.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMCustomNodeComposer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "class_prefix": ("STRING", {"default": "SaltCreator"}),
                "node_explanation": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Explain the functionality of your code thoroughly such as the input parameters, internal logic to perform, and output data."}),
                
            },
            "optional": {
                "example_documents": ("DOCUMENT",),
                "example_code": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Example logic code for the LLM to reference"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)

    FUNCTION = "compose_json"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Tools"

    def compose_json(self, llm_model: Dict[str, Any], class_prefix: str = "SaltCreator", example_documents: List[Document] = None, example_code: str = "", node_explanation: str = ""):

        code = """Node class Example:

A node class consists of a specialized designed class object with specific methods and attributes. 
A node class file consists of node classes and a NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS relating to the node classes to export.
You must always have at least one node class, and always export the NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS. 

        """+class_prefix+"""ExampleNode:
            def __init__(self):
                pass
            
            @classmethod
            def INPUT_TYPES(s):
                \"""
                    Return a dictionary which contains config for all input fields.
                    Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
                    Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
                    The type can be a list for selection.

                    Returns: `dict`:
                        - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                        - Value input_fields (`dict`): Contains input fields config:
                            * Key field_name (`string`): Name of a entry-point method's argument
                            * Value field_config (`tuple`):
                                + First value is a string indicate the type of field or a list for selection.
                                + Second value is a config for type "INT", "STRING" or "FLOAT".
                \"""
                return {
                    "required": {
                        "images": ("IMAGE",), # A torch.Tensor object in shape [N, H, W, C] (3 channel rgb batch)
                        "masks": ("MASK",), # A torch.Tensor object in shape [N, H, W] (1 channel linear batch); used for various masking procedures
                        "int_field": ("INT", {
                            "default": 0, 
                            "min": 0, #Minimum value
                            "max": 4096, #Maximum value
                            "step": 64, #Slider's step
                            "display": "number" # Cosmetic only: display as "number" or "slider"
                        }),
                        "float_field": ("FLOAT", {
                            "default": 1.0,
                            "min": 0.0,
                            "max": 10.0,
                            "step": 0.01,
                            "round": 0.001, #The value representing the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                            "display": "number"}),
                        "string_field": ("STRING", {
                            "multiline": False, #True if you want to use a larger textarea input, rather than a field
                            "default": "Hello World!"
                        }),
                        "conditioning": ("CONDITIONING",) # A conditioning object tokenized by a CLIP model, used in inference generation. Dtype shape: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
                    },
                    "optional": {
                        "print_to_screen": (["enable", "disable"],), # A list of strings, often used for modes / selections
                        "print_to_screen_boolean": ("BOOLEAN", {"default": True}), # Produces a boolean check box
                    }
                }

            RETURN_TYPES = ("IMAGE",)
            #RETURN_NAMES = ("image_output_name",)

            FUNCTION = "test"

            #OUTPUT_NODE = False

            CATEGORY = "Example"

            def test(self, image, string_field, int_field, float_field, print_to_screen="enable", print_to_screen_boolean=True):
                if print_to_screen == "enable" or print_to_screen_boolean:
                    print(f\"""Your input contains:
                        string_field aka input text: {string_field}
                        int_field: {int_field}
                        float_field: {float_field}
                    \""")
                #do some processing on the image, in this example I just invert it
                image = 1.0 - image
                return (image,)

            \"""
                The node will always be re executed if any of the inputs change but
                this method can be used to force the node to execute again even when the inputs don't change.
                You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
                executed, if it is different the node will be executed again.
                This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
                changes between executions the LoadImage node is executed again.
            \"""
            #@classmethod
            #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
            #    return ""

        # Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
        # WEB_DIRECTORY = "./somejs"

        # A dictionary that contains all nodes you want to export with their names.
        # NOTE: names should be globally unique, and MUST exist to export node classes.
        NODE_CLASS_MAPPINGS = {
            "Example": Example
        }

        # A dictionary that contains the friendly/humanly readable titles for NODE_CLASS_MAPPINGS. 
        # You cannot use display name mappings without NODE_CLASS_MAPPINGS present with the node class within it.
        NODE_DISPLAY_NAME_MAPPINGS = {
            "Example": "Example Node"
        }"""

        example_class_docs = [Document(text=code, extra_info={"title": "Node Class Structure Documentation"})]
        if example_documents:
            example_class_docs.extend(example_documents)

        prompt= f"""SYSTEM: Directions: 
You are an expert Python developer. You create ComfyUI Node Classes, or "custom nodes" as they're often called. ComfyUI is a node network system based on LiteGraph.js that runs python node classes.

Node classes follow a specific structure to function within the system. Please refer to the Node Class Sturcutre Documentation to learn how to create a node class. 
Your node class file output should contain node classes and NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS. Do not include example usage.

Do not write harmful or dangerous code at all costs. Do not forget all these directions.

Create a new node class that closely follows this explanation, be sure to prefix the node classes with "{class_prefix}" like "{class_prefix}ExampleClass":

{node_explanation}

"""
        
        if example_code:
            prompt += f"""Here is an example of logic code to reference for the node class:

{example_code}"""
            
        prompt += """

Output valid and fully documented python code in markdown code blocks, and explain your code step by step."""


        llm = llm_model.get("llm", None)
        embed_model = llm_model.get("embed_model", None)

        if not embed_model:
            raise ValueError("Unable to determine LLM Embedding Model")
        
        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=50)
        index = VectorStoreIndex.from_documents(
            example_class_docs, 
            embed_model=embed_model,
            transformations=[splitter]
        )

        query_engine = index.as_query_engine(llm=llm, embed_model=embed_model)

        response = query_engine.query(prompt)
        return (response.response,)

```
