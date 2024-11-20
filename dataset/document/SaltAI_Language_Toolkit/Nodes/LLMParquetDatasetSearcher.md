# ∞ Dataset/File Search (1-Dimensional)
## Documentation
- Class name: `LLMParquetDatasetSearcher`
- Category: `SALT/Language Toolkit/Tools/Dataset`
- Output node: `False`

The LLMParquetDatasetSearcher node is designed to efficiently search and filter data within Parquet files based on specified search terms, exclusion criteria, and length constraints. It leverages parallel processing to enhance search performance and incorporates relevancy scoring to prioritize results, making it a powerful tool for extracting and analyzing specific segments of large datasets.
## Input types
### Required
- **`file_type`**
    - Specifies the type of file to be searched. This parameter determines the appropriate reader to use for the file, supporting formats such as parquet, text, json, yaml, csv, and excel.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`path_or_url`**
    - The path to the file or URL to be searched. It is the starting point for the search operation, allowing the node to access and process the specified file.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`search_term`**
    - Defines the search terms used to filter the dataset. These terms are crucial for identifying relevant data segments within the file.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`exclude_terms`**
    - Terms to be excluded from the search results, refining the search by removing irrelevant or undesired data segments.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`columns`**
    - Specifies the columns to be included in the search. By default, all columns are included ('*'), but this can be customized to target specific columns.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`case_sensitive`**
    - Determines whether the search should be case sensitive, affecting how search terms are matched against the dataset.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`max_results`**
    - The maximum number of search results to return, allowing control over the volume of data retrieved.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`term_relevancy_threshold`**
    - The threshold for term relevancy scoring, used to prioritize results based on their relevance to the search terms.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`use_relevancy`**
    - Indicates whether relevancy scoring should be used to prioritize search results, enhancing the search's focus on the most relevant segments.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`min_length`**
    - The minimum length of the data segments to be included in the search results, enabling the exclusion of too-short segments.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`max_length`**
    - The maximum length of the data segments to be considered in the search results, preventing overly long segments from being included.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`max_dynamic_retries`**
    - The maximum number of retries with dynamic search terms if no results are found, enhancing the chance of retrieving relevant data.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`clean_content`**
    - Specifies whether the content should be cleaned, removing unnecessary or irrelevant information from the search results.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`excel_sheet_position`**
    - For Excel files, specifies the sheet to be searched by its position, allowing targeted searches within multi-sheet documents.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`recache`**
    - Indicates whether the file should be recached, potentially updating the cached version with the latest content.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`condense_documents`**
    - Determines whether documents should be condensed, potentially merging similar documents to reduce redundancy.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`seed`**
    - A seed value for random operations, ensuring reproducibility of the search results under the same conditions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`results`**
    - Comfy dtype: `STRING`
    - The primary search results, containing data segments that match the search criteria.
    - Python dtype: `List[Dict[str, Any]]`
- **`results_list`**
    - Comfy dtype: `LIST`
    - A list of search results, potentially including additional metadata or context about the matches.
    - Python dtype: `List[Dict[str, Any]]`
- **`documents`**
    - Comfy dtype: `DOCUMENT`
    - A collection of documents derived from the search results, formatted for further processing or analysis.
    - Python dtype: `List[Dict[str, Any]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMParquetDatasetSearcher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_type": (["parquet", "text", "json", "yaml", "csv", "excel"],),
                "path_or_url": ("STRING", {"placeholder": "Path to file or URL"}),
            },
            "optional": {
                "search_term": ("STRING", {"placeholder": "Enter search term"}),
                "exclude_terms": ("STRING", {"placeholder": "Terms to exclude, comma-separated"}),
                "columns": ("STRING", {"default": "*"}),
                "case_sensitive": ("BOOLEAN", {"default": False}),
                "max_results": ("INT", {"default": 10, "min": 1}),
                "term_relevancy_threshold": ("FLOAT", {"min": 0.0, "max": 1.0, "default": 0.25, "step": 0.01}),
                "use_relevancy": ("BOOLEAN", {"default": False}),
                #"num_threads": ("INT", {"default": 2}),
                "min_length": ("INT", {"min": 0, "max": 1023, "default": 0}),
                "max_length": ("INT", {"min": 3, "max": 1024, "default": 128}),
                "max_dynamic_retries": ("INT", {"default": 3}),
                "clean_content": ("BOOLEAN", {"default": False}),
                "excel_sheet_position": ("INT", {"min": 0, "default": "0"}),
                "recache": ("BOOLEAN", {"default": False}),
                "condense_documents": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING", "LIST", "DOCUMENT")
    RETURN_NAMES = ("results", "results_list", "documents")
    OUTPUT_IS_LIST = (True, False, False)

    FUNCTION = "search_dataset"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Tools/Dataset"

    def search_dataset(self, path_or_url, file_type, search_term="", exclude_terms="", columns="*", case_sensitive=False, max_results=10,
                       term_relevancy_threshold=None, use_relevancy=False, num_threads=2, min_length=0, max_length=-1, max_dynamic_retries=0,
                       clean_content=False, seed=None, excel_sheet_position="0", condense_documents=True, recache=False):
        
        # Validate path or download file and return path
        path = resolve_path(path_or_url)

        reader = ParquetReader1D()
        if file_type == "parquet":
            reader.from_parquet(path)
        elif file_type == "text":
            reader.from_text(path, recache=recache)
        elif file_type == "json":
            reader.from_json(path, recache=recache)
        elif file_type == "yaml":
            reader.from_yaml(path, recache=recache)
        elif file_type == "csv":
            reader.from_csv(path, recache=recache)
        elif file_type == "excel":
            reader.from_excel(path, sheet_name=excel_sheet_position, recache=recache)

        results = reader.search(
            search_term=search_term,
            exclude_terms=exclude_terms,
            columns=[col.strip() for col in columns.split(',') if col] if columns else ["*"],
            max_results=max_results,
            case_sensitive=case_sensitive,
            term_relevancy_score=term_relevancy_threshold,
            num_threads=num_threads,
            min_length=min_length,
            max_length=max_length,
            max_dynamic_retries=max_dynamic_retries,
            parse_content=clean_content,
            seed=min(seed, 99999999),
            use_relevancy=use_relevancy
        )

        results_list = []
        results_text = "Prompts:\n\n"
        documents = []
        for result in results:
            results_list.append(list(result.values())[0])
            if not condense_documents:
                documents.append(Document(text=list(result.values())[0], extra_info={}))
            else:
                results_text += str(list(result.values())[0]) + "\n\n"
        if condense_documents:
            documents = [Document(text=results_text, extra_info={})]

        return (results_list, results_list, documents,)

```
