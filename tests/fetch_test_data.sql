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
ORDER BY event_count;