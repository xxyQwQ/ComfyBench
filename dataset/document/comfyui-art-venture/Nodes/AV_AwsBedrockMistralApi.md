---
tags:
- LoRA
---

# AWS Bedrock Mistral API
## Documentation
- Class name: `AV_AwsBedrockMistralApi`
- Category: `ArtVenture/LLM`
- Output node: `False`

This node is designed to facilitate the creation of an API connection to AWS Bedrock Mistral, leveraging AWS credentials to authenticate and configure the API for use in language model applications.
## Input types
### Required
- **`aws_access_key_id`**
    - The AWS access key ID is essential for authenticating the user's AWS account, enabling secure access to AWS services.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`aws_secret_access_key`**
    - The AWS secret access key works alongside the access key ID to authenticate and secure the user's AWS account access.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`aws_session_token`**
    - The AWS session token provides temporary access credentials for using AWS services, enhancing security for short-term access.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`region`**
    - Specifies the AWS region for the API connection, determining the geographical location of the resources accessed.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`llm_api`**
    - Comfy dtype: `LLM_API`
    - Represents the configured API connection to AWS Bedrock Mistral, ready for use in language model applications.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AwsBedrockMistralApiNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aws_access_key_id": ("STRING", {"multiline": False}),
                "aws_secret_access_key": ("STRING", {"multiline": False}),
                "aws_session_token": ("STRING", {"multiline": False}),
                "region": (aws_regions, {"default": aws_regions[0]}),
            },
        }

    RETURN_TYPES = ("LLM_API",)
    RETURN_NAMES = ("llm_api",)
    FUNCTION = "create_api"
    CATEGORY = "ArtVenture/LLM"

    def create_api(self, aws_access_key_id, aws_secret_access_key, aws_session_token, region):
        if not aws_access_key_id or aws_access_key_id == "":
            aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", None)
        if not aws_secret_access_key or aws_secret_access_key == "":
            aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY", None)
        if not aws_session_token or aws_session_token == "":
            aws_session_token = os.environ.get("AWS_SESSION_TOKEN", None)

        if not aws_access_key_id or not aws_secret_access_key:
            raise Exception("AWS credentials is required.")

        return (
            AwsBedrockMistralApi(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region=region,
            ),
        )

```
