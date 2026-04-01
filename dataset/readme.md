# XCheck Dataset

To support reproducibility while respecting platform policies and copyright constraints, we release **XCheck** in a policy-compliant format containing only tweet IDs rather than redistributing raw media. Reconstructing full post content requires "hydration" via official X APIs or other compliant procedures. Any hydrated content remains subject to X's Terms of Service and the original creators' rights.

## Data Structure

Each data entry is a dictionary with the following format:

```python
{
    "id": "unique_identifier",          # Tweet ID (deceptive) or VisualNews ID (non-deceptive)
    "label": "label_value",             # "deceptive" or "non-deceptive"
    "LLM_topics": ["topic1", "topic2"], # Topical labels
    "LLM_factors": ["factor1"],         # Manipulated factors
    "summarized_evidence": "text"       # External context via reverse image search
}
```

### Notes:
- **Deceptive posts**: `id` corresponds to the tweet ID from X (formerly Twitter)
- **Non-deceptive posts**: `id` corresponds to the entry ID from the [VisualNews](https://github.com/FuxiaoLiu/VisualNews-Repository) dataset

## Hydration Instructions

To retrieve original post data (image, text, datetime), use the [X API](https://docs.x.com/x-api/introduction) with the provided tweet IDs. Please ensure compliance with X's API usage policies.