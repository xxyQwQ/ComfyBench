- `ImpactSEGSOrderedFilter`: The ImpactSEGSOrderedFilter node is designed to filter and order segmentation data based on specified criteria. It allows for the sorting of segments according to various attributes such as area, width, height, or coordinates, and enables the selection of a subset of these segments based on their order.
    - Parameters:
        - `target`: Specifies the attribute based on which the segments should be ordered. This can include attributes like area, width, height, or coordinates, impacting how the segments are sorted. Type should be `COMBO[STRING]`.
        - `order`: Determines the order in which the segments are sorted. A boolean value where True indicates descending order and False indicates ascending order. Type should be `BOOLEAN`.
        - `take_start`: Defines the starting index from which segments should be taken after ordering, allowing for the selection of a specific subset of segments. Type should be `INT`.
        - `take_count`: Specifies the number of segments to take starting from the 'take_start' index, enabling control over the size of the resulting segment subset. Type should be `INT`.
    - Inputs:
        - `segs`: The segmentation data to be filtered and ordered. This is the primary input over which the ordering and filtering operations are performed. Type should be `SEGS`.
    - Outputs:
        - `filtered_SEGS`: The segments that have been filtered and ordered according to the specified criteria. Type should be `SEGS`.
        - `remained_SEGS`: The segments that did not meet the filtering criteria and were not included in the ordered subset. Type should be `SEGS`.