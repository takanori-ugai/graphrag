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
 * retrieve the closest text units by vector similarity, and ask the chat model to answer
 * using those snippets as context.
 */
class BasicQueryEngine(
    private val chatModel: OpenAiChatModel,
    private val embeddingModel: EmbeddingModel,
    private val vectorStore: LocalVectorStore,
    private val textUnits: List<TextUnit>,
    private val textEmbeddings: List<TextEmbedding>,
    private val topK: Int = 5,
) {
    suspend fun answer(
        question: String,
        responseType: String,
    ): QueryResult {
        val queryEmbedding = embed(question) ?: return QueryResult("Failed to embed query text.", emptyList())
        val context = selectContext(queryEmbedding)
        val prompt = buildPrompt(question, responseType, context)
        val answer = generate(prompt)
        return QueryResult(answer = answer, context = context)
    }

    private suspend fun embed(text: String): List<Double>? =
        withContext(Dispatchers.IO) {
            runCatching {
                val response: Response<dev.langchain4j.data.embedding.Embedding> = embeddingModel.embed(text)
                response
                    .content()
                    ?.vector()
                    ?.asList()
                    ?.map { it.toDouble() }
            }.getOrNull()
        }

    private fun selectContext(queryEmbedding: List<Double>): List<QueryContextChunk> {
        // Prefer the persisted vector store for retrieval, otherwise fall back to in-memory embeddings.
        val nearest =
            vectorStore
                .nearestTextChunks(queryEmbedding, topK)
                .mapNotNull { (chunkId, distance) ->
                    textUnits.find { it.chunkId == chunkId }?.let { QueryContextChunk(chunkId, it.text, distance) }
                }

        if (nearest.isNotEmpty()) return nearest

        // Fallback: compute cosine similarity against embeddings stored in context.json.
        val byChunkId = textUnits.associateBy { it.chunkId }
        val scored =
            textEmbeddings
                .mapNotNull { embedding ->
                    val textUnit = byChunkId[embedding.chunkId] ?: return@mapNotNull null
                    val score = cosineSimilarity(queryEmbedding, embedding.vector)
                    QueryContextChunk(textUnit.chunkId, textUnit.text, score)
                }.sortedByDescending { it.score }
                .take(topK)

        return scored
    }

    private fun buildPrompt(
        question: String,
        responseType: String,
        context: List<QueryContextChunk>,
    ): String {
        val contextBlock =
            context.joinToString("\n\n") { chunk ->
                "- [${chunk.id}] ${chunk.text.take(800)}"
            }
        return """
            You are a helpful assistant that answers questions using the supplied context only.
            Provide the response in the form: $responseType.

            Context:
            $contextBlock

            Question: $question

            Answer:
            """.trimIndent()
    }

    private suspend fun generate(prompt: String): String =
        withContext(Dispatchers.IO) {
            runCatching { responder.answer(prompt) }.getOrNull() ?: "No response generated."
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

    private val responder: QueryResponder =
        AiServices.create(QueryResponder::class.java, chatModel)
}

private interface QueryResponder {
    @SystemMessage("You are a helpful assistant. Answer the question using only the provided context.")
    fun answer(
        @UserMessage prompt: String,
    ): String
}
