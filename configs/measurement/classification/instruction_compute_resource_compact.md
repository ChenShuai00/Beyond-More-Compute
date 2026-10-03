Classify the current paper window into compute-resource labels.

Output contract:
Return exactly one line containing only a valid JSON array of label strings.
Use double quotes only. Do not output markdown, explanation, prose, or a JSON object.
The first character must be [ and the last character must be ].
Never return [].

Valid labels are:
L1_INTERNAL, L2_CLOUD, L3_PUBLIC_HPC, L4_EXTERNAL_API, L5_OPEN_MODEL, L0_NO_SPECIFIC_RESOURCE.

Rules:
- Use only evidence in the current window.
- Select every supported L1-L5 label.
- If no L1-L5 label is supported, return only ["L0_NO_SPECIFIC_RESOURCE"].
- L0_NO_SPECIFIC_RESOURCE is mutually exclusive with L1-L5.
- Do not infer a resource from title, venue, affiliation, topic, or method name alone.
- Related work, citations, future work, generic acknowledgements, and generic discussion do not count unless the window shows author-side use.
- Few-shot examples are boundary examples only; do not copy labels unless the current window has matching evidence.
