package com.microsoft.graphrag.query

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonPrimitive

internal object JsonAnswerParser {
    data class Parsed(
        val raw: String,
        val response: String,
        val followUps: List<String>,
        val score: Double?,
    )

    fun parse(raw: String): Parsed {
        val fallback = Parsed(raw = raw, response = raw, followUps = emptyList(), score = null)
        return runCatching {
            val element = Json.parseToJsonElement(raw)
            val obj = element as? JsonObject ?: return fallback
            val response = obj["response"]?.jsonPrimitive?.content ?: raw
            val followUps =
                (obj["follow_up_queries"] as? JsonArray)
                    ?.mapNotNull { it.jsonPrimitive.contentOrNull }
                    .orEmpty()
            val score = obj["score"]?.jsonPrimitive?.doubleOrNull
            Parsed(raw = raw, response = response, followUps = followUps, score = score)
        }.getOrElse { fallback }
    }
}
