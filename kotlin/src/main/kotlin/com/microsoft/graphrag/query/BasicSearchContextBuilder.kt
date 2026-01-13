package com.microsoft.graphrag.query

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingType
import com.microsoft.graphrag.index.LocalVectorStore
import com.microsoft.graphrag.index.TextEmbedding
import com.microsoft.graphrag.index.TextUnit
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.output.Response
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.sqrt

/**
 * Basic search context builder that mirrors the Python BasicSearchContext: embeds the query, retrieves similar
 * text units, enforces a token budget, and returns both the rendered context text and structured records.
 */
class BasicSearchContextBuilder(
    private val embeddingModel: EmbeddingModel,
    private val vectorStore: LocalVectorStore,
    private val textUnits: List<TextUnit>,
    private val textEmbeddings: List<TextEmbedding>,
    private val columnDelimiter: String = "|",
    private val encoding: Encoding = Encodings.newLazyEncodingRegistry().getEncoding(EncodingType.CL100K_BASE),
    private val textIdColumn: String = "source_id",
    private val textColumn: String = "text",
) {
    private val logger = KotlinLogging.logger {}

    data class BasicContextResult(
        val contextText: String,
        val contextChunks: List<QueryContextChunk>,
        val contextRecords: Map<String, List<MutableMap<String, String>>>,
        val llmCalls: Int = 0,
        val promptTokens: Int = 0,
        val outputTokens: Int = 0,
        val llmCallsCategories: Map<String, Int> = emptyMap(),
        val promptTokensCategories: Map<String, Int> = emptyMap(),
        val outputTokensCategories: Map<String, Int> = emptyMap(),
    )

    @Suppress("CyclomaticComplexMethod")
    suspend fun buildContext(
        query: String,
        k: Int = 10,
        maxContextTokens: Int = 12_000,
        contextName: String = "Sources",
    ): BasicContextResult {
        val queryEmbedding = if (query.isBlank()) null else embed(query)
        val textByChunkId = textUnits.associateBy { it.chunkId }
        val nearest =
            if (queryEmbedding.isNullOrEmpty()) {
                emptyList()
            } else {
                vectorStore
                    .nearestTextChunks(queryEmbedding, k)
                    .mapNotNull { (chunkId, distance) ->
                        textByChunkId[chunkId]?.let { QueryContextChunk(it.id, it.text, distance) }
                    }
            }

        val fallback =
            if (nearest.isNotEmpty() || queryEmbedding.isNullOrEmpty()) {
                emptyList()
            } else {
                textEmbeddings
                    .mapNotNull { embedding ->
                        val unit = textByChunkId[embedding.chunkId] ?: return@mapNotNull null
                        val score = cosineSimilarity(queryEmbedding, embedding.vector)
                        QueryContextChunk(unit.id, unit.text, score)
                    }.sortedByDescending { it.score }
                    .take(k)
            }

        val ranked = if (nearest.isNotEmpty()) nearest else fallback
        val header = "$textIdColumn$columnDelimiter$textColumn"
        val rows = StringBuilder()
        val contextChunks = mutableListOf<QueryContextChunk>()
        val contextRecords = mutableListOf<MutableMap<String, String>>()
        var tokens = tokenCount("$header\n")

        for (chunk in ranked) {
            val row = "${chunk.id}$columnDelimiter${chunk.text}\n"
            val rowTokens = tokenCount(row)
            if (tokens + rowTokens > maxContextTokens) {
                break
            }
            rows.append(row)
            tokens += rowTokens
            contextChunks += chunk
            contextRecords += mutableMapOf(textIdColumn to chunk.id, textColumn to chunk.text)
        }

        val contextText =
            if (rows.isEmpty()) {
                "$header\n"
            } else {
                "$header\n${rows.toString().trimEnd()}"
            }

        return BasicContextResult(
            contextText = contextText,
            contextChunks = contextChunks,
            contextRecords = mapOf(contextName to contextRecords),
            llmCalls = 0,
            promptTokens = tokenCount(contextText),
            outputTokens = 0,
            llmCallsCategories = mapOf("build_context" to 0),
            promptTokensCategories = mapOf("build_context" to tokenCount(contextText)),
            outputTokensCategories = mapOf("build_context" to 0),
        )
    }

    private suspend fun embed(text: String): List<Double>? =
        withContext(Dispatchers.IO) {
            runCatching {
                val response: Response<dev.langchain4j.data.embedding.Embedding> = embeddingModel.embed(text)
                response
                    .content()
                    .vector()
                    .asList()
                    .map { it.toDouble() }
            }.getOrElse { error ->
                logger.warn { "Embedding failed for basic search ($error)" }
                null
            }
        }

    private fun tokenCount(text: String): Int = encoding.countTokens(text)

    private fun cosineSimilarity(
        a: List<Double>,
        b: List<Double>,
    ): Double {
        if (a.isEmpty() || b.isEmpty() || a.size != b.size) return 0.0
        var dot = 0.0
        var magA = 0.0
        var magB = 0.0
        for (i in a.indices) {
            dot += a[i] * b[i]
            magA += a[i] * a[i]
            magB += b[i] * b[i]
        }
        val denom = sqrt(magA) * sqrt(magB)
        return if (denom == 0.0) 0.0 else dot / denom
    }
}
