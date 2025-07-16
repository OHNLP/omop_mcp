MCP_DOC_INSTRUCTION = """
When selecting the best OMOP concept and vocabulary, always refer to the official OMOP CDM v5.4 documentation: https://ohdsi.github.io/CommonDataModel/faq.html and https://ohdsi.github.io/CommonDataModel/vocabulary.html.
Use the mapping conventions, standard concept definitions, and vocabulary guidance provided there to ensure your selection is accurate and consistent with OMOP best practices. Prefer concepts that are marked as 'Standard' and 'Valid', and use the recommended vocabularies for each domain (e.g., SNOMED for conditions, RxNorm for drugs, LOINC for measurements, etc.) unless otherwise specified.

**You must return ONLY the following fields, in this exact order, with no extra formatting, no markdown, no bullets, and no additional text. Do NOT use markdown, bullets, or add explanations outside the REASON field.**

Return mapping result using ALL fields in this exact format, with each field on a new line:
CONCEPT_ID: ...
CODE: ...
NAME: ...
CLASS: ...
CONCEPT: ...
VALIDITY: ...
DOMAIN: ...
VOCAB: ...
URL: ...
REASON: ...

IMPORTANT: The URL field must be the raw URL only, do not format it as a markdown link or add any brackets.
For the REASON field, provide a concise explanation of why this concept was selected, any special considerations about the mapping, and how additional details from the source term should be handled in OMOP.

**Do NOT do this:**
- **CONCEPT_ID**: 12345
- **CODE**: 67890
(bullets, markdown, or extra text are not allowed)

**Again, only output the fields as shown above, with no extra formatting or text.**
""".strip()

EXAMPLE_INPUT = "Map `Temperature Temporal Scanner - RR` for `measurement_concept_id` in the `measurement` table."

EXAMPLE_OUTPUT = """CONCEPT_ID: 46235152
CODE: 75539-7
NAME: Body temperature - Temporal artery
CLASS: Clinical Observation
CONCEPT: Standard
VALIDITY: Valid
DOMAIN: Measurement
VOCAB: LOINC
URL: https://athena.ohdsi.org/search-terms/terms/46235152
REASON: This LOINC concept specifically represents body temperature measured at the temporal artery, which is what a temporal scanner measures. The \"RR\" in your source term likely refers to \"Recovery Room\" or another location/department indicator, but in OMOP, the location would typically be captured in a separate field rather than as part of the measurement concept itself."""
