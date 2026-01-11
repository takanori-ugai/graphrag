package com.microsoft.graphrag.query

import com.microsoft.graphrag.index.LocalVectorStore
import com.microsoft.graphrag.index.TextEmbedding
import com.microsoft.graphrag.index.TextUnit
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.model.output.Response
import dev.langchain4j.service.AiServices
import dev.langchain4j.service.SystemMessage
import dev.langchain4j.service.UserMessage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Locale
import kotlin.math.sqrt

data class QueryContextChunk(
    val id: String,
    val text: String,
    val score: Double,
)

data class QueryResult(
    val answer: String,
    val context: List<QueryContextChunk>,
)

/**
 * Minimal query engine that mirrors the Python basic search flow: embed the question,
 * retrieve the closest text units by vector similarity, build a CSV context table with
 * a token budget, and ask the chat model to answer using the basic search system prompt.
 */
@Suppress("LongParameterList")
class BasicQueryEngine(
    private val chatModel: OpenAiChatModel,
    private val embeddingModel: EmbeddingModel,
    private val vectorStore: LocalVectorStore,
    private val textUnits: List<TextUnit>,
    private val textEmbeddings: List<TextEmbedding>,
    private val topK: Int = 10,
    private val maxContextTokens: Int = 12_000,
    private val columnDelimiter: String = "|",
) {
    suspend fun answer(
        question: String,
        responseType: String,
    ): QueryResult {
        val queryEmbedding = embed(question) ?: return QueryResult("Failed to embed query text.", emptyList())
        val contextRows = buildContext(queryEmbedding)
        val prompt = buildPrompt(responseType, contextRows)
        val answer = generate(prompt, question)
        return QueryResult(answer = answer, context = contextRows)
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
            }.getOrNull()
        }

    private fun buildContext(queryEmbedding: List<Double>): List<QueryContextChunk> {
        // Prefer the persisted vector store for retrieval, otherwise fall back to in-memory embeddings.
        val nearest =
            vectorStore
                .nearestTextChunks(queryEmbedding, topK)
                .mapNotNull { (chunkId, distance) ->
                    textUnits.find { it.chunkId == chunkId }?.let { QueryContextChunk(it.id, it.text, distance) }
                }

        if (nearest.isNotEmpty()) return nearest

        // Fallback: compute cosine similarity against embeddings stored in context.json.
        val byChunkId = textUnits.associateBy { it.chunkId }
        val scored =
            textEmbeddings
                .mapNotNull { embedding ->
                    val textUnit = byChunkId[embedding.chunkId] ?: return@mapNotNull null
                    val score = cosineSimilarity(queryEmbedding, embedding.vector)
                    QueryContextChunk(textUnit.id, textUnit.text, score)
                }.sortedByDescending { it.score }
                .take(topK)

        val headerTokens = tokenCount("source_id${columnDelimiter}text\n")
        var tokens = headerTokens
        val rows = mutableListOf<QueryContextChunk>()
        for (chunk in scored) {
            val rowText = "${chunk.id}$columnDelimiter${chunk.text}\n"
            val rowTokens = tokenCount(rowText)
            if (tokens + rowTokens > maxContextTokens) {
                break
            }
            tokens += rowTokens
            rows.add(chunk)
        }
        return rows
    }

    private fun buildPrompt(
        responseType: String,
        context: List<QueryContextChunk>,
    ): String {
        val header = "source_id$columnDelimiter${"text"}"
        val rows =
            context.joinToString("\n") { chunk ->
                "${chunk.id}$columnDelimiter${chunk.text}"
            }
        val contextBlock = "$header\n$rows"
        val systemPrompt =
            BASIC_SEARCH_SYSTEM_PROMPT
                .replace("{context_data}", contextBlock)
                .replace("{response_type}", responseType)
        return systemPrompt
    }

    private suspend fun generate(
        systemPrompt: String,
        question: String,
    ): String =
        withContext(Dispatchers.IO) {
            val finalPrompt = "$systemPrompt\n\nUser question: $question"
            runCatching { responder.answer(finalPrompt) }.getOrNull() ?: "No response generated."
        }

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

    private fun tokenCount(text: String): Int =
        text
            .lowercase(Locale.US)
            .split(Regex("\\s+"))
            .count { it.isNotBlank() }

    private val responder: QueryResponder =
        AiServices.create(QueryResponder::class.java, chatModel)
}

private interface QueryResponder {
    @SystemMessage("You are a helpful assistant. Answer the question using only the provided context.")
    fun answer(
        @UserMessage prompt: String,
    ): String
}

private val BASIC_SEARCH_SYSTEM_PROMPT =
    """
---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all relevant information in the input data tables appropriate for the response length and format.

You should use the data provided in the data tables below as the primary context for generating the response.

If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: Sources (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Sources (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Sources (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the source id taken from the "source_id" column in the provided tables.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}


---Data tables---

{context_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all relevant information in the input data appropriate for the response length and format.

You should use the data provided in the data tables below as the primary context for generating the response.

If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: Sources (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Sources (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Sources (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the source id taken from the "source_id" column in the provided tables.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
    """.trimIndent()
