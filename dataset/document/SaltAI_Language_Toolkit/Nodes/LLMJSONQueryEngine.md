# ∞ JSON Query Engine
## Documentation
- Class name: `LLMJSONQueryEngine`
- Category: `SALT/Language Toolkit/Querying`
- Output node: `False`

This node is designed to process queries against JSON data using a language model, enabling the extraction of information or execution of tasks based on a JSON schema and data. It leverages a JSON query engine to interpret and respond to queries, making it suitable for applications requiring dynamic data interpretation or manipulation.
## Input types
### Required
- **`llm_model`**
    - Specifies the language model to be used for processing queries. This model is crucial for understanding and generating responses based on the JSON data and schema provided.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `Dict[str, Any]`
- **`json_data`**
    - The actual JSON data on which queries will be executed. This data is parsed according to the provided JSON schema.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`json_schema`**
    - Defines the structure of the JSON data to be queried. This schema guides the query engine in interpreting the data correctly.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`prompt`**
    - The query or command to be executed against the JSON data. This prompt drives the interaction with the data.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`human_readable`**
    - A flag indicating whether the response should be synthesized for human readability. This affects how the query results are presented.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`optional_service_context`**
    - An optional context that can be provided to customize the service's behavior or processing environment.
    - Comfy dtype: `LLM_CONTEXT`
    - Python dtype: `Dict[str, Any]`
## Output types
- **`result`**
    - Comfy dtype: `STRING`
    - The direct response from the query execution, containing the information or results extracted from the JSON data.
    - Python dtype: `str`
- **`json_path`**
    - Comfy dtype: `STRING`
    - Additional data about the query execution, including paths within the JSON data that were accessed to generate the response.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMJSONQueryEngine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "json_data": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Enter JSON data here..."}),
                "json_schema": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Enter JSON schema here..."}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Enter your JSON query / question here..."}),
                "human_readable": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "optional_service_context": ("LLM_CONTEXT",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result", "json_path")

    FUNCTION = "query_engine"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Querying"

    def query_engine(self, llm_model, json_schema, json_data, prompt, human_readable, optional_service_context=None):
        try:
            schema = json.loads(json_schema)
            data = json.loads(json_data)
        except json.JSONDecodeError as e:
            error_message = f"JSON parsing error: {str(e)}. Please ensure your JSON schema and data are correctly formatted."
            logger.error(error_message)
            return ("", "")

        query_engine = JSONQueryEngine(
            json_value=data,
            json_schema=schema,
            llm=llm_model['llm'],
            synthesize_response=True if human_readable else False,
            service_context=optional_service_context if optional_service_context else None
        )

        response = query_engine.query(prompt)
        logger.data(response, indent=4)

        return (response.response, response.metadata["json_path_response_str"])

```
