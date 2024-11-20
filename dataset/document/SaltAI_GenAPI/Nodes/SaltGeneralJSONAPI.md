# General API JSON Request
## Documentation
- Class name: `SaltGeneralJSONAPI`
- Category: `SALT/API`
- Output node: `False`

The SaltGeneralJSONAPI node is designed to facilitate the creation and handling of JSON-based requests for general APIs. It abstracts the complexities involved in preparing requests, including sanitization, handling files, and setting headers, thereby streamlining the process of interacting with various web services through a unified interface.
## Input types
### Required
- **`preset`**
    - A preset configuration for the API request, allowing for predefined settings to be applied automatically.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `Tuple[List[str]]`
- **`api_base`**
    - The base URL for the API request, serving as the foundation to which endpoint paths and parameters are appended.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`optional_user_agent`**
    - An optional user agent string to be included in the request headers, identifying the client to the server.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`optional_api_key`**
    - An optional API key for authentication, included in the request headers to access restricted endpoints or features.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`request`**
    - The JSON-formatted string representing the request to be sent. It is sanitized and placeholders are replaced with actual values before sending, ensuring that the request is properly formatted and contains all necessary information.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`a`**
    - A wildcard placeholder for dynamic content that can be included in the request, offering flexibility in the request composition.
    - Comfy dtype: `*`
    - Python dtype: `str`
- **`b`**
    - A wildcard placeholder for dynamic content, potentially representing images or other binary data, to be included in the request.
    - Comfy dtype: `*`
    - Python dtype: `str`
- **`c`**
    - A wildcard placeholder for additional dynamic content, enhancing the request's flexibility and customization.
    - Comfy dtype: `*`
    - Python dtype: `str`
- **`d`**
    - A wildcard placeholder for further dynamic content, allowing for a wide range of data to be included in the request.
    - Comfy dtype: `*`
    - Python dtype: `str`
- **`e`**
    - A wildcard placeholder for extra dynamic content, ensuring maximum flexibility in the request's composition.
    - Comfy dtype: `*`
    - Python dtype: `str`
## Output types
- **`response`**
    - Comfy dtype: `*`
    - The raw response from the API, encapsulating the outcome of the request.
    - Python dtype: `str`
- **`json`**
    - Comfy dtype: `STRING`
    - The JSON-parsed response, providing structured access to the data returned by the API.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltGeneralJSONAPI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": (["Salt AI Ollama",],), # Mock Placeholder for Presets
                "api_base": ("STRING", {"default": "http://35.193.61.202:11434/api/generate"}), #placeholder address, need JS presets
                "optional_user_agent": ("STRING", {"default": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}),
                "optional_api_key": ("STRING", {"default": ""}),
                "request": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": """{
    "model": "llava",
    "prompt": "{a}",
    "images": "{b}",
    "format": "json",
    "stream": false,
    "options": {
        "temperature": 0.7
    }
}"""}), #placeholder default
            },
            "optional": {
                "a": (WILDCARD,),
                "b": (WILDCARD,),
                "c": (WILDCARD,),
                "d": (WILDCARD,),
                "e": (WILDCARD,),
            }
        }

    RETURN_TYPES = (WILDCARD, "STRING")
    RETURN_NAMES = ("response", "json")

    FUNCTION = "dispatch"
    CATEGORY = f"{MENU_NAME}/API"

    def handle_files(self, request_dict):
        files = {}
        for key, value in request_dict.items():
            if isinstance(value, str) and os.path.isfile(value):
                files[key] = (os.path.basename(value), open(value, 'rb'))
                request_dict[key] = None
        return files

    def dispatch(self, preset, api_base, optional_user_agent, optional_api_key, request, **kwargs):
        try:
            request = sanitize_json(request)
            request_dict = json.loads(request)
        except json.JSONDecodeError as e:
            raise Exception(f"Error in initial JSON formatting: {e}")

        request_dict = replace_placeholders(request_dict, kwargs)

        files = self.handle_files(request_dict)
        if files:
            request_dict = {k: v for k, v in request_dict.items() if v is not None}
            params = {k: (str(v) if not isinstance(v, tuple) else v) for k, v in {**request_dict, **files}.items()}
            mp_encoder = MultipartEncoder(fields=params)
            headers = {
                "Content-Type": mp_encoder.content_type
            }
            data = mp_encoder
        else:
            headers = {
                "Content-Type": "application/json"
            }
            data = json.dumps(request_dict)

        if optional_user_agent.strip():
            headers["User-Agent"] = optional_user_agent

        if optional_api_key.strip():
            headers["Authorization"] = f"Bearer {optional_api_key}"

        logger.info("Original Request:")
        logger.data(request, indent=4)
        logger.info("Processed Request JSON:")
        logger.data(request_dict, indent=4)
        logger.info("Headers: ")
        logger.data(headers, indent=4)

        try:
            response = requests.post(api_base, data=data, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request Exception: {e}")
            logger.error(f"Response Content: {e.response.content if e.response else 'No response content'}")
            raise Exception(f"Error: {e}")
        except Exception as e:
            logger.error(f"Unknown Error:")
            logger.error(str(e))
            raise Exception(e)

        logger.info("Response: ")
        logger.data(response.text, indent=4)

        return (response.text, json.dumps(request_dict))

```
