# ∞ Generate Reply
## Documentation
- Class name: `GenerateReply`
- Category: `SALT/Language Toolkit/Agents/Chat`
- Output node: `False`

This node facilitates the generation of replies in a chat context by processing messages sent to and from agents. It leverages conversational agents to simulate interactive dialogues, dynamically generating responses based on the input message and the context of the conversation.
## Input types
### Required
- **`recipient`**
    - Specifies the agent intended to receive the message, playing a crucial role in determining the context and content of the generated reply.
    - Comfy dtype: `AGENT`
    - Python dtype: `ConversableAgent`
- **`message`**
    - The message content sent to the recipient, which is used as the basis for generating a reply. Its content directly influences the reply's relevance and coherence.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`sender`**
    - Identifies the sender of the message, providing additional context for the reply generation process. It's optional and defaults to None if not provided.
    - Comfy dtype: `AGENT`
    - Python dtype: `ConversableAgent`
## Output types
- **`reply`**
    - Comfy dtype: `STRING`
    - The generated reply from the recipient agent, based on the input message and the conversation context.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class GenerateReply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "recipient": ("AGENT",),
                "message": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            },
            "optional": {
                "sender": ("AGENT", {"default": None}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("reply",)

    FUNCTION = "start_chat"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Agents/Chat"

    def start_chat(self, recipient, message, sender=None):
        recipient = clone_conversable_agent(recipient)
        if sender:
            sender = clone_conversable_agent(sender)
        message = recipient._oai_messages[sender] + [{"content": message, "role": "user"}]
        reply = recipient.generate_reply(message, sender)
        return (reply,)

```
