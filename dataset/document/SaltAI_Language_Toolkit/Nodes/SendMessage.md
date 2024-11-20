# ∞ Send Message
## Documentation
- Class name: `SendMessage`
- Category: `SALT/Language Toolkit/Agents/Chat`
- Output node: `False`

The SendMessage node facilitates communication between agents by allowing one agent to send a message to another. It abstracts the complexities of agent interaction, ensuring messages are delivered and processed correctly.
## Input types
### Required
- **`recipient`**
    - Specifies the agent receiving the message. It plays a crucial role in determining the message's destination.
    - Comfy dtype: `AGENT`
    - Python dtype: `AGENT`
- **`sender`**
    - Identifies the agent sending the message. It is essential for tracking the source of the message and for potential reply mechanisms.
    - Comfy dtype: `AGENT`
    - Python dtype: `AGENT`
- **`message`**
    - Contains the text of the message being sent. This parameter is vital for the communication process, allowing agents to exchange information.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`recipient`**
    - Comfy dtype: `AGENT`
    - The recipient agent, potentially modified after receiving the message.
    - Python dtype: `AGENT`
- **`sender`**
    - Comfy dtype: `AGENT`
    - The sender agent, potentially modified after sending the message.
    - Python dtype: `AGENT`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SendMessage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "recipient": ("AGENT",),
                "sender": ("AGENT",),
                "message": ("STRING", {
                    "multiline": True,
                    "default": "Hi"
                }),
            },
        }

    RETURN_TYPES = ("AGENT", "AGENT")
    RETURN_NAMES = ("recipient", "sender")

    FUNCTION = "add_info"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Agents/Chat"

    def add_info(self, recipient, sender, message):
        recipient = clone_conversable_agent(recipient)
        sender = clone_conversable_agent(sender)
        sender.send(message, recipient)
        return (recipient, sender)

```
