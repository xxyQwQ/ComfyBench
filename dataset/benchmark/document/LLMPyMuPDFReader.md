- `LLMPyMuPDFReader`: The LLMPyMuPDFReader node is designed to read PDF files and convert them into a llama_index Document format, leveraging the PyMuPDF library for efficient processing. This node facilitates the extraction of text and potentially metadata from PDF documents, making them accessible for further analysis or processing within the llama_index ecosystem.
    - Inputs:
        - `path` (Required): Specifies the file path to the PDF document to be read. This path is essential for locating and accessing the file for processing. Type should be `STRING`.
        - `metadata` (Required): A boolean flag indicating whether metadata should be extracted from the PDF document alongside the text. This option allows for more comprehensive document analysis by including additional information. Type should be `COMBO[BOOLEAN]`.
        - `extra_info` (Optional): A string containing extra configuration or information in JSON format, which can be used to customize the reading process. This parameter allows for flexible adaptation to specific requirements. Type should be `STRING`.
    - Outputs:
        - `documents`: The output is a Document object containing the extracted text (and optionally metadata) from the PDF file, ready for integration into the llama_index ecosystem. Type should be `DOCUMENT`.
