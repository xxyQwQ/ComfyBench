# ∞ Documents Chat Engine
## Documentation
- Class name: `LLMChatEngine`
- Category: `SALT/Language Toolkit/Querying`
- Output node: `False`

The LLMChatEngine node facilitates interactive chat sessions by initializing and managing a chat engine based on a specified language model index. It supports dynamic conversation flow by allowing the reset of the chat engine and processing user queries to generate responses.
## Input types
### Required
- **`llm_index`**
    - The index of the language model to be used for the chat session. It determines which chat engine instance is initialized for handling queries.
    - Comfy dtype: `LLM_INDEX`
    - Python dtype: `LLM_INDEX`
- **`query`**
    - The user's query as a string. It is the input for the chat engine to process and generate a response.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`reset_engine`**
    - A boolean flag indicating whether to reset the chat engine before processing the current query. This allows for starting a new conversation context.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The generated response from the chat engine as a string, based on the user's query.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMChatEngine:
        def __init__(self):
            self.chat_engine = None
            
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "llm_index": ("LLM_INDEX",),
                    "query": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Ask a question"}),
                },
                "optional": {
                    "reset_engine": ("BOOLEAN", {"default": False})
                }
            }

        RETURN_TYPES = ("STRING",)
        FUNCTION = "chat"
        CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Querying"

        def chat(self, llm_index, query:str, reset_engine:bool = False) -> str:
            if not self.chat_engine or reset_engine:
                self.chat_engine = llm_index.as_chat_engine()
            response = self.chat_engine.chat(query)
            return (response.response,)

```
