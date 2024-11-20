# ∞ Message
## Documentation
- Class name: `LLMChatMessages`
- Category: `SALT/Language Toolkit/Messages`
- Output node: `False`

The LLMChatMessagesAdv node is designed to prepare and structure chat messages for interaction between a system and a user. It encapsulates the process of converting raw text inputs into a formatted list of chat messages, facilitating a structured dialogue flow.
## Input types
### Required
- **`prompt`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
- **`role`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
## Output types
- **`llm_message`**
    - Comfy dtype: `LIST`
    - A list of structured chat messages, representing the dialogue between the system and the user. This output facilitates further processing or interaction within a chat-based application.
    - Python dtype: `List[ChatMessage]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMChatMessages:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "prompt"}),
                "role": (["SYSTEM", "USER"],),
            },
        }

    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("llm_message", )

    FUNCTION = "prepare_messages"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Messages"

    def prepare_messages(self, prompt, role):
        messages = [
                ChatMessage(role=MessageRole.SYSTEM if role == "SYSTEM" else MessageRole.USER, content=repr(prompt) ),
        ]
        return (messages,)

```
