- `LLMImageTextReader`: The LLMImageTextReader node is designed to read and process images, extracting text and additional information based on user-defined parameters. It leverages underlying image processing and text extraction technologies to facilitate the analysis and interpretation of image content.
    - Parameters:
        - `path`: Specifies the file path of the image to be processed. This is a crucial parameter as it determines the source image for text extraction and further processing. Type should be `STRING`.
        - `parse_text`: A boolean flag indicating whether text should be extracted from the image. This affects whether the node will perform text parsing operations on the image. Type should be `COMBO[BOOLEAN]`.
        - `extra_info`: A string containing extra information in JSON format that can be used to provide additional context or instructions for processing the image. Type should be `STRING`.
    - Inputs:
    - Outputs:
        - `documents`: The output is a document containing the results of the image processing and text extraction, structured in a format that can be further analyzed or displayed. Type should be `DOCUMENT`.