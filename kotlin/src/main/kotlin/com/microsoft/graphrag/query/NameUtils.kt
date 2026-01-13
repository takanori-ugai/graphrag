package com.microsoft.graphrag.query

internal fun sanitizeName(name: String): String = name.trim().ifBlank { "index" }.replace("\\s+".toRegex(), "_")
