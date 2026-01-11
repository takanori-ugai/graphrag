package com.microsoft.graphrag.query

/**
 * Simple model parameter holder to mirror Python config knobs.
 * Currently used to toggle JSON-only responses and carry tuning values.
 */
data class ModelParams(
    val temperature: Double? = null,
    val topP: Double? = null,
    val maxTokens: Int? = null,
    val jsonResponse: Boolean = false,
)
