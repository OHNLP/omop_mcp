MCP_DOC_INSTRUCTION = """
When selecting the best OMOP concept and vocabulary, ALWAYS check the omop://documentation resource first for official OMOP CDM v5.4 vocabulary rules and mapping guidelines.

Prefer concepts that are marked as 'Standard' and 'Valid'. When selecting the best OMOP concept and vocabulary, 

**IMPORTANT: Before making any vocabulary decisions, access omop://documentation and omop://preferred_vocabularies to see the current guidelines and preferences.**

**USER PREFERENCE HANDLING:**
- If user says "Find from LOINC vocabulary" → ONLY consider LOINC concepts
- If user says "Prefer SNOMED" → Prioritize SNOMED concepts
- If user specifies a vocabulary → That vocabulary becomes the PRIMARY choice
- Default preferences only apply when user doesn't specify a preference

The find_omop_concept tool will return multiple candidate concepts with their metadata. You must evaluate and select the most appropriate concept based on:

1. **Clinical Appropriateness**: Does the concept accurately represent the clinical term?
2. **Context Requirements**: Any specific vocabulary, validity, or other requirements mentioned in the prompt
3. **OMOP Best Practices**: Generally prefer Standard + Valid concepts from preferred vocabularies
4. **Use Case Considerations**: Research needs, granularity requirements, etc.

**IMPORTANT: You are not limited to Standard/Valid concepts if the context requires otherwise (e.g., mapping legacy data, specific vocabulary requirements, research needs).**

**You must return ONLY the following fields, in this exact order, with no extra formatting, no markdown, no bullets, and no additional text. Do NOT use markdown, bullets, or add explanations outside the REASON field.**

**After using any tool, you must always provide a REASON field. In the REASON, explain:**
- **How you interpreted the keyword **
- **Why you selected the specific OMOP concept from the available candidates**

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
For the REASON field, provide a detailed explanation of your selection process, why this concept was chosen over other candidates, any special considerations about the mapping, and how additional details from the source term should be handled in OMOP.

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
