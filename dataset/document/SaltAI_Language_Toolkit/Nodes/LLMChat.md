# ∞ Multi Query
## Documentation
- Class name: `LLMChat`
- Category: `SALT/Language Toolkit/Querying`
- Output node: `False`

The LLMChat node facilitates interactive chat sessions with a language model, allowing users to submit queries and receive responses. It dynamically initializes or resets the chat engine based on the session requirements, ensuring tailored interactions.
## Input types
### Required
- **`llm_model`**
    - Specifies the language model configuration for the chat session, including any necessary model details for initialization.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `Dict[str, Any]`
### Optional
- **`llm_context`**
    - Provides context or additional settings for the language model, potentially influencing the chat session's behavior.
    - Comfy dtype: `LLM_CONTEXT`
    - Python dtype: `Any`
- **`llm_message`**
    - A list of messages that have been part of the chat session, used to maintain context or continuity in the conversation.
    - Comfy dtype: `LIST`
    - Python dtype: `List[ChatMessage]`
- **`llm_documents`**
    - A list of documents that can be referenced by the language model during the chat session, enriching the responses.
    - Comfy dtype: `DOCUMENT`
    - Python dtype: `List[Any]`
- **`prompt`**
    - The user's input or question to the language model, serving as the basis for generating a response.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`response`**
    - Comfy dtype: `STRING`
    - The language model's response to the user's query, encapsulating the interaction's outcome.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMChat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL", ),
            },
            "optional": {
                "llm_context": ("LLM_CONTEXT", ),
                "llm_message": ("LIST", {}),
                "llm_documents": ("DOCUMENT", {}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("response", )

    FUNCTION = "chat"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Querying"

    def chat(self, llm_model:Dict[str, Any], prompt:str, llm_context:Any = None, llm_message:List[ChatMessage] = None, llm_documents:List[Any] = None) -> str:

        embed_model = llm_model.get('embed_model', None)

        if not prompt and not llm_message:
            err = "A `prompt` or `llm_message` must be provided for querying"
            logger.error(err)
            raise ValueError(err)

        # Spoof documents -- Why can't we just talk to a modeL?
        if not llm_documents:
            logger.info("Using blank documents.")
            documents = [Document(text="null", extra_info={})]
        else:
            logger.info("Using provided documents")
            documents = llm_documents

        logger.data(documents)

        index = VectorStoreIndex.from_documents(
            documents, 
            embed_model=embed_model,
            service_context=llm_context,
            transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=20)]
        )
        
        if not llm_message:
            llm_message = [ChatMessage(role=MessageRole.USER, text="")]

        if not prompt:
            prompt = "null"

        template = ChatPromptTemplate(message_templates=llm_message)
        query_engine = index.as_query_engine(llm=llm_model.get("llm", None), embed_model=embed_model, text_qa_template=template)
        response = query_engine.query(prompt)
        return (response.response, )

```
