package com.microsoft.graphrag.query

/**
 * Produces a sanitized identifier from an input name.
 *
 * Trims leading and trailing whitespace, substitutes `"index"` when the trimmed name is empty, and replaces each sequence of whitespace characters with a single underscore.
 *
 * @param name Input name to sanitize.
 * @return The sanitized name with normalized whitespace and guaranteed non-empty content.
 */
internal fun sanitizeName(name: String): String = name.trim().ifBlank { "index" }.replace("\\s+".toRegex(), "_")