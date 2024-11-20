---
tags:
- ImpactPack
- Segmentation
---

# SEGS Classify
## Documentation
- Class name: `ImpactSEGSClassify`
- Category: `ImpactPack/HuggingFace`
- Output node: `False`

The ImpactSEGSClassify node is designed to classify segments (SEGS) based on specified criteria, comparing attributes or labels within each segment to filter and categorize them accordingly. This process involves evaluating conditions such as greater than, less than, or equal to, between attributes or label scores within the segments, facilitating advanced segmentation and classification tasks.
## Input types
### Required
- **`classifier`**
    - The 'classifier' input specifies the classification model to be used for evaluating the segments. It plays a crucial role in determining the classification outcomes based on the model's capabilities.
    - Comfy dtype: `TRANSFORMERS_CLASSIFIER`
    - Python dtype: `Any`
- **`segs`**
    - The 'segs' input represents the segments to be classified, serving as the primary data upon which classification conditions are applied. It is crucial for determining which segments meet the specified criteria and which do not, affecting the node's execution and results.
    - Comfy dtype: `SEGS`
    - Python dtype: `List[Tuple[Any, Any]]`
- **`preset_expr`**
    - The 'preset_expr' input allows for the selection of predefined classification expressions or the use of a manual expression. It defines the criteria for classification, impacting the filtering process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`manual_expr`**
    - The 'manual_expr' input is used when a custom classification expression is required, providing flexibility in defining specific conditions for segment classification.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`ref_image_opt`**
    - The 'ref_image_opt' input is an optional parameter that specifies a reference image from which segments may be extracted or compared, enhancing the classification accuracy.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Optional[Any]`
## Output types
- **`filtered_SEGS`**
    - Comfy dtype: `SEGS`
    - The 'filtered_SEGS' output contains segments that meet the specified classification criteria, effectively separating them based on the comparison conditions applied.
    - Python dtype: `List[Any]`
- **`remained_SEGS`**
    - Comfy dtype: `SEGS`
    - The 'remained_SEGS' output includes segments that do not meet the classification criteria, allowing for further processing or exclusion from certain analyses.
    - Python dtype: `List[Any]`
- **`detected_labels`**
    - Comfy dtype: `STRING`
    - The 'detected_labels' output lists all unique labels identified during the classification process, offering insights into the variety of labels present within the segments.
    - Python dtype: `List[str]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SEGS_Classify:
    @classmethod
    def INPUT_TYPES(s):
        global preset_classify_expr
        return {"required": {
                        "classifier": ("TRANSFORMERS_CLASSIFIER",),
                        "segs": ("SEGS",),
                        "preset_expr": (preset_classify_expr + ['Manual expr'],),
                        "manual_expr": ("STRING", {"multiline": False}),
                     },
                "optional": {
                     "ref_image_opt": ("IMAGE", ),
                    }
                }

    RETURN_TYPES = ("SEGS", "SEGS", "STRING")
    RETURN_NAMES = ("filtered_SEGS", "remained_SEGS", "detected_labels")
    OUTPUT_IS_LIST = (False, False, True)

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/HuggingFace"

    @staticmethod
    def lookup_classified_label_score(score_infos, label):
        global symbolic_label_map

        if label.startswith('#'):
            if label not in symbolic_label_map:
                return None
            else:
                label = symbolic_label_map[label]
        else:
            label = {label}

        for x in score_infos:
            if x['label'] in label:
                return x['score']

        return None

    def doit(self, classifier, segs, preset_expr, manual_expr, ref_image_opt=None):
        if preset_expr == 'Manual expr':
            expr_str = manual_expr
        else:
            expr_str = preset_expr

        match = re.match(classify_expr_pattern, expr_str)

        if match is None:
            return (segs[0], []), segs, []

        a = match.group(1)
        op = match.group(2)
        b = match.group(3)

        a_is_lab = not is_numeric_string(a)
        b_is_lab = not is_numeric_string(b)

        classified = []
        remained_SEGS = []
        provided_labels = set()

        for seg in segs[1]:
            cropped_image = None

            if seg.cropped_image is not None:
                cropped_image = seg.cropped_image
            elif ref_image_opt is not None:
                # take from original image
                cropped_image = crop_image(ref_image_opt, seg.crop_region)

            if cropped_image is not None:
                cropped_image = to_pil(cropped_image)
                res = classifier(cropped_image)
                classified.append((seg, res))

                for x in res:
                    provided_labels.add(x['label'])
            else:
                remained_SEGS.append(seg)

        filtered_SEGS = []
        for seg, res in classified:
            if a_is_lab:
                avalue = SEGS_Classify.lookup_classified_label_score(res, a)
            else:
                avalue = a

            if b_is_lab:
                bvalue = SEGS_Classify.lookup_classified_label_score(res, b)
            else:
                bvalue = b

            if avalue is None or bvalue is None:
                remained_SEGS.append(seg)
                continue

            avalue = float(avalue)
            bvalue = float(bvalue)

            if op == '>':
                cond = avalue > bvalue
            elif op == '<':
                cond = avalue < bvalue
            elif op == '>=':
                cond = avalue >= bvalue
            elif op == '<=':
                cond = avalue <= bvalue
            else:
                cond = avalue == bvalue

            if cond:
                filtered_SEGS.append(seg)
            else:
                remained_SEGS.append(seg)

        return (segs[0], filtered_SEGS), (segs[0], remained_SEGS), list(provided_labels)

```
