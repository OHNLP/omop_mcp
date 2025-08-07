-- FROM MHH database

-- Find measurement keywords
SELECT clinical_event, COUNT(clinical_event) as event_count
FROM CLINICAL_EVENTS
WHERE unit <> ''
GROUP BY clinical_event
ORDER BY event_count DESC;

-- Find procedure keywords
SELECT [procedure], COUNT([procedure]) as procedure_count
FROM SURGERY
WHERE [procedure] <> ''
GROUP BY [procedure]
ORDER BY procedure_count DESC;

-- Find condition keywords
SELECT source_string, COUNT(source_string) as event_count, source_identifier, source
FROM DIAGNOSIS
WHERE source_string <> ''
GROUP BY source_string, source_identifier, source
ORDER BY event_count DESC;

-- Find medication keywords
SELECT order_mnemonic as medication, COUNT(order_mnemonic) AS event_count
FROM MED_ADMIN
WHERE order_mnemonic <> ''
GROUP BY order_mnemonic
ORDER BY event_count DESC;

-- FROM UTHealth OMOP

-- Get mapping test datasets from UTPhysicians
-- Find measurement keywords
-- Sources: CLARITY, ALLSCRIPTS, SUNRISE_HCPC
SELECT
    DISTINCT CONCAT(UPPER(SRC_PANEL_NAME), ' - ', UPPER(SRC_COMPONENT_NAME)) AS keyword,
    NULL AS count,
    STRING_AGG(CAST(athena_id AS VARCHAR), ', ') AS concept_id_manual_mapping
FROM mappings.master_lab_mappings_index
WHERE athena_id <> ''
    AND SRC <> 'MHH_COVID'

-- Find procedure keywords with concatenated concept IDs
SELECT
    SRC_NAME as keyword,
    SUM(FREQ) AS count,
    STRING_AGG(CAST(procedure_concept_id AS VARCHAR), ', ') AS concept_id_manual_mapping
FROM mappings.master_procedure_mappings_index
WHERE procedure_concept_id <> ''
    AND SRC <> 'MHH_COVID'
GROUP BY SRC_NAME
ORDER BY count DESC;

-- Find medication keywords
SELECT DISTINCT SRC_CODE AS keyword,
       NULL AS count,
       CODE AS concept_id_manual_mapping
FROM mappings.master_drug_mappings_index
WHERE CODE <> '' AND SRC <> 'MHH_COVID';

