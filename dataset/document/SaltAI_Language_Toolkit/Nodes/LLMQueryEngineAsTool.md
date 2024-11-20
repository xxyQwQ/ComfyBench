# ∞ Query Engine as Tool
## Documentation
- Class name: `LLMQueryEngineAsTool`
- Category: `SALT/Language Toolkit/Agents/Tools`
- Output node: `False`

This node encapsulates a query engine as a tool, enabling the execution of queries against a document to extract or generate information based on the input question. It is designed to facilitate communication with documents by analyzing their content and providing relevant answers.
## Input types
### Required
- **`name`**
    - Specifies the name of the tool, used for identification and display purposes.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`description`**
    - Provides a detailed description of the tool's functionality and its intended use case.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`llm_index`**
    - Identifies the language model index to be used for querying, enabling the tool to access the appropriate resources for information retrieval.
    - Comfy dtype: `LLM_INDEX`
    - Python dtype: `custom type, specific to the implementation`
## Output types
- **`query_tool`**
    - Comfy dtype: `TOOL`
    - Returns a tool configured to perform queries, encapsulating the query engine functionality within a callable interface.
    - Python dtype: `dict`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMQueryEngineAsTool:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"multiline": False, "dynamicPrompts": False, "placeholder": "code"}),
                "description": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "A function that allows you to communicate with a document. Ask a question and this function will find information in the document and generate an answer."}),
                "llm_index": ("LLM_INDEX",),
            },
        }

    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ("query_tool",)

    FUNCTION = "return_tool"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Agents/Tools"
    
    def return_tool(self, name, description, llm_index):
        def query_engine(query: str) -> str:
            query_components = []
            query_components.append("Analyze the above document carefully to find your answer. If you can't find one, say so.")

            if query:
                if query.strip():
                    query_components.append("user: " + query)
            query_components.append("assistant:")
            query_join = "\n".join(query_components)

            query_engine = llm_index.as_query_engine()
            response = query_engine.query(query_join)
            return (response.response,)
        tool = {"name": name, "description": description, "function": query_engine}
        return (tool,)

```
