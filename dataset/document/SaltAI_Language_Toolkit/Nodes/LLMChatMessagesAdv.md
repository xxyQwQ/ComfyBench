# ∞ Message (Advanced)
## Documentation
- Class name: `LLMChatMessagesAdv`
- Category: `SALT/Language Toolkit/Messages`
- Output node: `False`

The LLMChatMessagesAdv node is designed to prepare chat messages by encapsulating system and user prompts into a structured format. This node plays a crucial role in initializing and structuring conversation data for further processing or interaction within a chat-based application.
## Input types
### Required
- **`system_prompt`**
    - The system prompt represents the initial message or context provided by the system, setting the stage for the user's response. It is crucial for defining the scenario or environment in which the user interaction takes place.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`user_prompt`**
    - The user prompt captures the user's input or response to the system's initial message. It is essential for driving the conversation forward and determining the direction of the interaction.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`llm_message`**
    - Comfy dtype: `LIST`
    - A list of structured chat messages that include both the system's and the user's inputs, ready for further processing or analysis.
    - Python dtype: `List[ChatMessage]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMChatMessagesAdv:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "You are a dog, you cannot speak, only woof, and react as a dog would."}),
                "user_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "plaeholder": "What is your name?"}),
            },
        }

    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("llm_message", )

    FUNCTION = "prepare_messages"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Messages"

    def prepare_messages(self, system_prompt, user_prompt):
        messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=repr(system_prompt) ),
                ChatMessage(role=MessageRole.USER, content=repr(user_prompt) ),
        ]
        return (messages,)

```
