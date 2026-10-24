"""Shared, code-defined SQL fragments for persistence contracts."""

HOUSE_PARSE_SOURCE = """
    CASE
        WHEN LOWER(COALESCE(parse_run.parser_version, '')) LIKE '%gemini%'
          OR LOWER(COALESCE(parse_run.parser_version, '')) LIKE '%ling%'
          OR LOWER(COALESCE(parse_run.parser_version, '')) LIKE '%gpt%'
          OR LOWER(COALESCE(parse_run.parser_version, '')) LIKE '%openrouter%'
          OR LOWER(COALESCE(parse_run.engines_attempted, '')) LIKE 'gemini/%'
          OR LOWER(COALESCE(parse_run.engines_attempted, '')) LIKE 'openrouter/%'
          OR LOWER(COALESCE(parse_run.engines_attempted, '')) LIKE 'gpt-%'
          OR LOWER(COALESCE(parse_run.engines_attempted, '')) LIKE 'gemini+%'
          OR LOWER(COALESCE(parse_run.engines_attempted, '')) LIKE 'visionb%'
        THEN 'gemini_ocr'
        ELSE 'house_pdf'
    END
"""
