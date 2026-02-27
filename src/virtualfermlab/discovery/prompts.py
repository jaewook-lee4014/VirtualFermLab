"""LLM prompt templates for kinetic parameter extraction."""

from __future__ import annotations

EXTRACTION_SYSTEM_PROMPT = (
    "You are a bioprocess engineering expert specialising in microbial "
    "GROWTH kinetics and fermentation modelling (Monod, Contois, Logistic, Andrews). "
    "Your task is to extract ORGANISM GROWTH parameters from scientific papers. "
    "CRITICAL: Only extract parameters that describe whole-cell microbial growth "
    "(e.g. maximum specific growth rate of the organism). "
    "Do NOT extract enzyme kinetic parameters (e.g. Km or Vmax of a purified enzyme, "
    "protease activity, lipase activity, cellulase kinetics). "
    "Do NOT extract parameters for chemical reactions, adsorption, or non-biological processes. "
    "Respond with valid JSON only. No markdown, no explanation, no code fences."
)

# ------------------------------------------------------------------
# Shared schema block (used in both prompt variants)
# ------------------------------------------------------------------

_JSON_SCHEMA_BLOCK = """\
The JSON schema is:
{{
  "strain_name": "<exact organism/strain name from the paper>",
  "kinetic_model": "<mathematical model used: Monod | Contois | Logistic | Andrews | Haldane | Tessier | Moser | empirical | null>",
  "parameters": [
    {{
      "name": "<param>",
      "value": <number>,
      "unit": "<unit>",
      "substrate": "<substrate or null>",
      "evidence": "<exact sentence or table row from the text where this value appears>"
    }}
  ],
  "conditions": {{
    "temperature": <number or null>,
    "pH": <number or null>,
    "fermentation_mode": "<batch | fed-batch | continuous | SSF | null>",
    "substrate_type": "<glucose | xylose | glucose+xylose | starch | cellulose | etc. or null>",
    "initial_substrate_conc_g_L": <number or null>,
    "glucose_xylose_ratio": "<e.g. 1:1 or 2:1 or null>",
    "inoculum_conc": "<number with unit, e.g. 0.1 g/L or 0.05 OD600 or null>",
    "reactor_type": "<flask | shake_flask | bioreactor | microplate | null>",
    "fermentation_duration_h": <number or null>,
    "final_biomass_g_L": <number or null>
  }}
}}

Rules for "kinetic_model":
- This is the mathematical model used to describe ORGANISM GROWTH, not enzyme kinetics.
- Common models: "Monod" (mu = mu_max * S / (Ks + S)), "Contois", "Logistic", "Andrews" or "Haldane" (substrate inhibition), "Tessier", "Moser".
- If the paper fits data to a specific model, report its name.
- If the paper reports parameters without naming a model, use "empirical".
- If unclear, use null.

Rules for "evidence":
- Copy the EXACT sentence or table row from the input text that contains the numeric value.
- Do NOT paraphrase or generate new text. It must be a verbatim quote.
- For table values, copy the relevant row as "column1: val1, column2: val2, ...".

Allowed parameter names (ORGANISM GROWTH parameters only):
- "mu_max": maximum specific growth rate of the organism (unit: "1/h")
- "Ks": substrate saturation/affinity constant for growth (unit: "g/L")
- "Yxs": biomass yield on substrate (unit: "g/g")
- "K_I": substrate inhibition constant for growth (unit: "g/L")
- "lag_time": lag phase duration (unit: "h")

DO NOT extract these (they are NOT growth kinetic parameters):
- Enzyme Km, Vmax, Ki (for purified enzymes like protease, lipase, cellulase, xylanase)
- pH_opt of an enzyme (e.g. "optimal pH for protease activity was 7.0")
- Activation energy, adsorption constants, partition coefficients
- Product formation kinetics (ethanol yield, lactic acid production rate)
- Parameters from chemical/physical models (not microbial growth)

HOW TO DISTINGUISH growth kinetics from enzyme kinetics:
- Growth mu_max: describes how fast the ORGANISM grows (units: 1/h or h^-1), \
measured by OD600, cell dry weight, biomass increase over time
- Enzyme Km: describes how an ISOLATED ENZYME binds substrate (units: mM or mg/mL), \
measured by enzyme assay, spectrophotometry on purified/crude enzyme
- Growth Ks: substrate concentration at which organism grows at half mu_max (units: g/L), \
fitted from growth curves
- Enzyme Ki: inhibitor concentration that reduces enzyme activity (units: mM), \
from enzyme inhibition assays"""

# ------------------------------------------------------------------
# Few-shot examples
# ------------------------------------------------------------------

_FEWSHOT_GROWTH = """\

### Example 1 — Growth kinetics from a Results section (EXTRACT these):

Input text:
"The maximum specific growth rate of Fusarium venenatum on glucose was \
determined to be 0.18 h\\u207b\\u00b9 using the Monod model (Ks = 0.5 g/L) \
with a biomass yield of 0.42 g/g. Batch fermentation was conducted at \
pH 6.0 and 28 \\u00b0C for 120 h in a 2 L bioreactor. Initial glucose \
concentration was 20 g/L and final biomass reached 6.8 g/L."

Expected output:
{{"strain_name": "Fusarium venenatum", "kinetic_model": "Monod", "parameters": [\
{{"name": "mu_max", "value": 0.18, "unit": "1/h", "substrate": "glucose", \
"evidence": "The maximum specific growth rate of Fusarium venenatum on glucose was determined to be 0.18 h\\u207b\\u00b9 using the Monod model"}}, \
{{"name": "Ks", "value": 0.5, "unit": "g/L", "substrate": "glucose", \
"evidence": "Ks = 0.5 g/L"}}, \
{{"name": "Yxs", "value": 0.42, "unit": "g/g", "substrate": "glucose", \
"evidence": "a biomass yield of 0.42 g/g"}}], \
"conditions": {{"temperature": 28, "pH": 6.0, "fermentation_mode": "batch", \
"substrate_type": "glucose", "initial_substrate_conc_g_L": 20, \
"glucose_xylose_ratio": null, "inoculum_conc": null, \
"reactor_type": "bioreactor", "fermentation_duration_h": 120, \
"final_biomass_g_L": 6.8}}}}"""

_FEWSHOT_TABLE = """\

### Example 2 — Table with kinetic parameters (EXTRACT these):

Input text:
"Table 2. Monod kinetic parameters for A. niger on different substrates
| Substrate | \\u03bc_max (h\\u207b\\u00b9) | Ks (g/L) | Yxs (g/g) |
| --- | --- | --- | --- |
| glucose | 0.25 | 0.15 | 0.45 |
| xylose  | 0.12 | 0.30 | 0.38 |
Cultivation: 30\\u00b0C, pH 5.5, shake flask, 72 h."

Expected output:
{{"strain_name": "A. niger", "kinetic_model": "Monod", "parameters": [\
{{"name": "mu_max", "value": 0.25, "unit": "1/h", "substrate": "glucose", \
"evidence": "Substrate: glucose, \\u03bc_max: 0.25 h\\u207b\\u00b9, Ks: 0.15 g/L, Yxs: 0.45 g/g"}}, \
{{"name": "Ks", "value": 0.15, "unit": "g/L", "substrate": "glucose", \
"evidence": "Substrate: glucose, \\u03bc_max: 0.25 h\\u207b\\u00b9, Ks: 0.15 g/L, Yxs: 0.45 g/g"}}, \
{{"name": "Yxs", "value": 0.45, "unit": "g/g", "substrate": "glucose", \
"evidence": "Substrate: glucose, \\u03bc_max: 0.25 h\\u207b\\u00b9, Ks: 0.15 g/L, Yxs: 0.45 g/g"}}, \
{{"name": "mu_max", "value": 0.12, "unit": "1/h", "substrate": "xylose", \
"evidence": "Substrate: xylose, \\u03bc_max: 0.12 h\\u207b\\u00b9, Ks: 0.30 g/L, Yxs: 0.38 g/g"}}, \
{{"name": "Ks", "value": 0.30, "unit": "g/L", "substrate": "xylose", \
"evidence": "Substrate: xylose, \\u03bc_max: 0.12 h\\u207b\\u00b9, Ks: 0.30 g/L, Yxs: 0.38 g/g"}}, \
{{"name": "Yxs", "value": 0.38, "unit": "g/g", "substrate": "xylose", \
"evidence": "Substrate: xylose, \\u03bc_max: 0.12 h\\u207b\\u00b9, Ks: 0.30 g/L, Yxs: 0.38 g/g"}}], \
"conditions": {{"temperature": 30, "pH": 5.5, "fermentation_mode": "batch", \
"substrate_type": "glucose+xylose", "initial_substrate_conc_g_L": null, \
"glucose_xylose_ratio": null, "inoculum_conc": null, \
"reactor_type": "shake_flask", "fermentation_duration_h": 72, \
"final_biomass_g_L": null}}}}"""

_FEWSHOT_NEGATIVE = """\

### Example 3 — Enzyme kinetics (DO NOT extract):

Input text:
"The purified protease from Aspergillus niger showed optimal activity at pH 7.0. \
Kinetic analysis revealed Km = 2.5 mM and Vmax = 150 U/mg using casein as substrate. \
Ki for PMSF was 0.8 mM."

Expected output (empty parameters — these are enzyme kinetics, NOT growth kinetics):
{{"strain_name": "Aspergillus niger", "kinetic_model": null, "parameters": [], \
"conditions": {{"temperature": null, "pH": null, "fermentation_mode": null, \
"substrate_type": null, "initial_substrate_conc_g_L": null, \
"glucose_xylose_ratio": null, "inoculum_conc": null, \
"reactor_type": null, "fermentation_duration_h": null, \
"final_biomass_g_L": null}}}}"""

_FEWSHOT_ABSTRACT = """\

### Example — Growth kinetics from an abstract:

Input text:
Title: "Batch fermentation kinetics of Rhizopus oryzae on glucose"
Abstract: "Rhizopus oryzae was cultivated in batch mode at 30 \\u00b0C and pH 5.0 \
on 15 g/L glucose for 96 h. The Monod model was fitted to growth data, yielding \
\\u03bcmax = 0.14 h\\u207b\\u00b9 and Ks = 0.8 g/L. The biomass yield was 0.40 g/g \
with a final biomass of 4.2 g/L."

Expected output:
{{"strain_name": "Rhizopus oryzae", "kinetic_model": "Monod", "parameters": [\
{{"name": "mu_max", "value": 0.14, "unit": "1/h", "substrate": "glucose", \
"evidence": "\\u03bcmax = 0.14 h\\u207b\\u00b9"}}, \
{{"name": "Ks", "value": 0.8, "unit": "g/L", "substrate": "glucose", \
"evidence": "Ks = 0.8 g/L"}}, \
{{"name": "Yxs", "value": 0.40, "unit": "g/g", "substrate": "glucose", \
"evidence": "The biomass yield was 0.40 g/g"}}], \
"conditions": {{"temperature": 30, "pH": 5.0, "fermentation_mode": "batch", \
"substrate_type": "glucose", "initial_substrate_conc_g_L": 15, \
"glucose_xylose_ratio": null, "inoculum_conc": null, \
"reactor_type": null, "fermentation_duration_h": 96, \
"final_biomass_g_L": 4.2}}}}"""

# ------------------------------------------------------------------
# Full-text prompt (used when Results/Tables text is available)
# ------------------------------------------------------------------

EXTRACTION_PROMPT_FULLTEXT = """\
Extract ORGANISM GROWTH kinetic parameters from the following paper.

Title: {title}

Full text (Results, Discussion, Tables):
{full_text}

IMPORTANT RULES:
- ONLY extract parameters for ORGANISM/CELL GROWTH (mu_max, Ks, Yxs, K_I, lag_time).
- Do NOT extract enzyme kinetic parameters (Km, Vmax, Ki of purified enzymes).
- Do NOT extract product formation parameters (ethanol yield, acid production rate).
- If the paper only discusses enzyme characterisation with no growth kinetics, return an empty parameters list.
- Only extract numeric values that are EXPLICITLY stated in the text.
- If a parameter is not mentioned, do NOT include it.
- "value" must be a number (int or float), never a string.
- When a table lists parameters under multiple conditions (e.g. different pH or substrates), \
extract each row as a separate parameter entry.
- For each parameter, include "evidence": the exact sentence or table row where the value appears.
- Identify which kinetic model was used (Monod, Contois, Logistic, Andrews, etc.).
- Extract ALL experimental conditions: temperature, pH, substrate type, initial concentration, \
fermentation mode, reactor type, duration, and final biomass.
- Return ONLY a single JSON object, nothing else.

""" + _JSON_SCHEMA_BLOCK + _FEWSHOT_GROWTH + _FEWSHOT_TABLE + _FEWSHOT_NEGATIVE + """

JSON:"""

# ------------------------------------------------------------------
# Abstract-only prompt (fallback when full-text is unavailable)
# ------------------------------------------------------------------

EXTRACTION_PROMPT_ABSTRACT = """\
Extract ORGANISM GROWTH kinetic parameters from the following paper.

Title: {title}
Abstract: {abstract}

IMPORTANT RULES:
- ONLY extract parameters for ORGANISM/CELL GROWTH (mu_max, Ks, Yxs, K_I, lag_time).
- Do NOT extract enzyme kinetic parameters (Km, Vmax, Ki of purified enzymes).
- If the abstract only discusses enzyme characterisation, return an empty parameters list.
- Only extract numeric values that are EXPLICITLY stated in the text.
- If a parameter is not mentioned, do NOT include it.
- If no numeric growth parameter values are explicitly stated, return an empty parameters list.
- "value" must be a number (int or float), never a string.
- For each parameter, include "evidence": the exact sentence where the value appears.
- Identify which kinetic model was used (Monod, Contois, Logistic, etc.) if mentioned.
- Extract experimental conditions: temperature, pH, substrate, mode, etc.
- Return ONLY a single JSON object, nothing else.

""" + _JSON_SCHEMA_BLOCK + _FEWSHOT_ABSTRACT + _FEWSHOT_NEGATIVE + """

JSON:"""

# Keep backward-compatible alias
EXTRACTION_PROMPT_TEMPLATE = EXTRACTION_PROMPT_ABSTRACT
