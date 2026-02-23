"""LLM prompt templates for kinetic parameter extraction."""

from __future__ import annotations

EXTRACTION_SYSTEM_PROMPT = (
    "You are a bioprocess engineering expert specialising in microbial "
    "growth kinetics and fermentation modelling. "
    "Respond with valid JSON only. No markdown, no explanation, no code fences, no reasoning."
)

# ------------------------------------------------------------------
# Shared schema block (used in both prompt variants)
# ------------------------------------------------------------------

_JSON_SCHEMA_BLOCK = """\
The JSON schema is:
{{
  "strain_name": "<exact organism/strain name from the paper>",
  "substrates": ["<substrate1>", "<substrate2>"],
  "parameters": [
    {{
      "name": "<param>",
      "value": <number>,
      "unit": "<unit>",
      "substrate": "<substrate or null>",
      "evidence": "<exact sentence or table row from the text where this value appears>"
    }}
  ],
  "conditions": {{"pH": <number or null>, "temperature": <number or null>, "mode": "<batch|fed-batch|continuous or null>"}}
}}

Rules for "evidence":
- Copy the EXACT sentence or table row from the input text that contains the numeric value.
- Do NOT paraphrase or generate new text. It must be a verbatim quote.
- For table values, copy the relevant row as "column1: val1, column2: val2, ...".

Allowed parameter names:
- "mu_max": maximum specific growth rate (unit: "1/h")
- "Ks": substrate saturation constant (unit: "g/L")
- "Yxs": biomass yield on substrate (unit: "g/g")
- "K_I": substrate inhibition constant (unit: "g/L")
- "pH_opt": optimal pH for growth (unit: "")
- "pH_min": minimum pH for growth (unit: "")
- "pH_max": maximum pH for growth (unit: "")
- "lag_time": lag phase duration (unit: "h")"""

# ------------------------------------------------------------------
# Few-shot examples
# ------------------------------------------------------------------

_FEWSHOT_RESULTS_SECTION = """\

### Example 1 — extracting from a Results section:

Input text:
\"The maximum specific growth rate was determined to be 0.25 h\u207b\u00b9 on glucose \
(Ks = 0.15 g/L) with a biomass yield of 0.45 g/g. The organism was \
Candida utilis grown at pH 5.5 and 30 \u00b0C in batch mode.\"

Expected output:
{{"strain_name": "Candida utilis", "substrates": ["glucose"], "parameters": [\
{{"name": "mu_max", "value": 0.25, "unit": "1/h", "substrate": "glucose", \
"evidence": "The maximum specific growth rate was determined to be 0.25 h\u207b\u00b9 on glucose"}}, \
{{"name": "Ks", "value": 0.15, "unit": "g/L", "substrate": "glucose", \
"evidence": "Ks = 0.15 g/L"}}, \
{{"name": "Yxs", "value": 0.45, "unit": "g/g", "substrate": "glucose", \
"evidence": "a biomass yield of 0.45 g/g"}}], \
"conditions": {{"pH": 5.5, "temperature": 30, "mode": "batch"}}}}"""

_FEWSHOT_TABLE = """\

### Example 2 — extracting from a Table:

Input text:
\"Table 2. Kinetic parameters for S. cerevisiae at different pH values
| pH  | \u03bc_max (h\u207b\u00b9) | Ks (g/L) |
| --- | ---          | ---      |
| 5.0 | 0.18         | 0.22     |
| 6.0 | 0.25         | 0.15     |\"

Expected output:
{{"strain_name": "S. cerevisiae", "substrates": [], "parameters": [\
{{"name": "mu_max", "value": 0.18, "unit": "1/h", "substrate": null, \
"evidence": "pH: 5.0, \u03bc_max: 0.18 h\u207b\u00b9, Ks: 0.22 g/L"}}, \
{{"name": "Ks", "value": 0.22, "unit": "g/L", "substrate": null, \
"evidence": "pH: 5.0, \u03bc_max: 0.18 h\u207b\u00b9, Ks: 0.22 g/L"}}, \
{{"name": "mu_max", "value": 0.25, "unit": "1/h", "substrate": null, \
"evidence": "pH: 6.0, \u03bc_max: 0.25 h\u207b\u00b9, Ks: 0.15 g/L"}}, \
{{"name": "Ks", "value": 0.15, "unit": "g/L", "substrate": null, \
"evidence": "pH: 6.0, \u03bc_max: 0.25 h\u207b\u00b9, Ks: 0.15 g/L"}}], \
"conditions": {{"pH": null, "temperature": null, "mode": null}}}}"""

_FEWSHOT_ABSTRACT = """\

### Example — extracting from an abstract:

Input text:
Title: \"Growth kinetics of Pichia pastoris on methanol\"
Abstract: \"Pichia pastoris was cultivated in fed-batch mode at 30 \u00b0C. \
The maximum specific growth rate on methanol was 0.14 h\u207b\u00b9 with a \
biomass yield of 0.40 g/g.\"

Expected output:
{{"strain_name": "Pichia pastoris", "substrates": ["methanol"], "parameters": [\
{{"name": "mu_max", "value": 0.14, "unit": "1/h", "substrate": "methanol", \
"evidence": "The maximum specific growth rate on methanol was 0.14 h\u207b\u00b9"}}, \
{{"name": "Yxs", "value": 0.40, "unit": "g/g", "substrate": "methanol", \
"evidence": "a biomass yield of 0.40 g/g"}}], \
"conditions": {{"pH": null, "temperature": 30, "mode": "fed-batch"}}}}"""

# ------------------------------------------------------------------
# Full-text prompt (used when Results/Tables text is available)
# ------------------------------------------------------------------

EXTRACTION_PROMPT_FULLTEXT = """\
Extract microbial growth kinetic parameters from the following paper.

Title: {title}

Full text (Results, Discussion, Tables):
{full_text}

Rules:
- Only extract numeric values that are EXPLICITLY stated in the text.
- If a parameter is not mentioned, do NOT include it.
- "value" must be a number (int or float), never a string.
- When a table lists parameters under multiple conditions (e.g. different pH or substrates), \
extract each row as a separate parameter entry.
- For each parameter, include "evidence": the exact sentence or table row where the value appears.
- Return ONLY a single JSON object, nothing else.

""" + _JSON_SCHEMA_BLOCK + _FEWSHOT_RESULTS_SECTION + _FEWSHOT_TABLE + """

JSON:"""

# ------------------------------------------------------------------
# Abstract-only prompt (fallback when full-text is unavailable)
# ------------------------------------------------------------------

EXTRACTION_PROMPT_ABSTRACT = """\
Extract microbial growth kinetic parameters from the following paper.

Title: {title}
Abstract: {abstract}

Rules:
- Only extract numeric values that are EXPLICITLY stated in the text.
- If a parameter is not mentioned, do NOT include it.
- If no numeric parameter values are explicitly stated, return an empty parameters list.
- "value" must be a number (int or float), never a string.
- For each parameter, include "evidence": the exact sentence where the value appears.
- Return ONLY a single JSON object, nothing else.

""" + _JSON_SCHEMA_BLOCK + _FEWSHOT_ABSTRACT + """

JSON:"""

# Keep backward-compatible alias
EXTRACTION_PROMPT_TEMPLATE = EXTRACTION_PROMPT_ABSTRACT
