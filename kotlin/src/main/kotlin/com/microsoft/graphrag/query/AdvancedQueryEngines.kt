package com.microsoft.graphrag.query

import com.microsoft.graphrag.index.CommunityReport
import com.microsoft.graphrag.index.Entity
import com.microsoft.graphrag.index.EntitySummary
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

/**
 * Local search mirrors the Python local search by prioritizing entity-centric
 * context (entity summaries if available, otherwise source chunks). It falls back
 * to text-unit similarity if no entity vectors are present.
 */
@Suppress("LongParameterList")
class LocalQueryEngine(
    private val chatModel: OpenAiChatModel,
    private val embeddingModel: EmbeddingModel,
    private val vectorStore: LocalVectorStore,
    private val textUnits: List<TextUnit>,
    private val textEmbeddings: List<TextEmbedding>,
    private val entities: List<Entity>,
    private val entitySummaries: List<EntitySummary>,
    private val topKEntities: Int = 5,
    private val fallbackTopK: Int = 5,
    private val maxContextChars: Int = 800,
) {
    suspend fun answer(
        question: String,
        responseType: String,
    ): QueryResult {
        val context = buildContext(question)
        val prompt = buildPrompt(question, responseType, context)
        val answer = generate(prompt)
        return QueryResult(answer = answer, context = context)
    }

    @Suppress("ReturnCount")
    suspend fun buildContext(question: String): List<QueryContextChunk> {
        val queryEmbedding = embed(question) ?: return emptyList()
        val entityContexts = selectEntityContext(queryEmbedding)
        if (entityContexts.isNotEmpty()) return entityContexts
        return selectTextContext(queryEmbedding)
    }

    private fun selectEntityContext(queryEmbedding: List<Double>): List<QueryContextChunk> {
        val summariesById = entitySummaries.associateBy { it.entityId }
        val entitiesById = entities.associateBy { it.id }
        return vectorStore
            .nearestEntities(queryEmbedding, topKEntities)
            .mapNotNull { (entityId, distance) ->
                val text =
                    summariesById[entityId]?.summary
                        ?: entitiesById[entityId]?.let { entity ->
                            textUnits.find { it.chunkId == entity.sourceChunkId }?.text
                        }
                text?.let { QueryContextChunk(id = entityId, text = it, score = distance) }
            }
    }

    private fun selectTextContext(queryEmbedding: List<Double>): List<QueryContextChunk> {
        val byChunkId = textUnits.associateBy { it.chunkId }
        return textEmbeddings
            .mapNotNull { embedding ->
                val textUnit = byChunkId[embedding.chunkId] ?: return@mapNotNull null
                val score = cosineSimilarity(queryEmbedding, embedding.vector)
                QueryContextChunk(id = textUnit.chunkId, text = textUnit.text, score = score)
            }.sortedByDescending { it.score }
            .take(fallbackTopK)
    }

    private fun buildPrompt(
        question: String,
        responseType: String,
        context: List<QueryContextChunk>,
    ): String {
        val contextBlock =
            context.joinToString("\n\n") { chunk ->
                "- [${chunk.id}] ${chunk.text.take(maxContextChars)}"
            }
        return """
            You are a helpful assistant answering a question using local (entity-level) context from a knowledge graph.
            Provide the response in the form: $responseType.

            Context:
            $contextBlock

            Question: $question

            Answer:
            """.trimIndent()
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

    private val responder: ContextResponder =
        AiServices.create(ContextResponder::class.java, chatModel)
}

/**
 * Global search leans on community reports; it embeds the summaries and selects
 * the most relevant communities as context.
 */
class GlobalQueryEngine(
    private val chatModel: OpenAiChatModel,
    private val embeddingModel: EmbeddingModel,
    private val communityReports: List<CommunityReport>,
    private val topK: Int = 3,
    private val maxContextChars: Int = 1000,
) {
    suspend fun answer(
        question: String,
        responseType: String,
    ): QueryResult {
        val context = buildContext(question)
        val prompt = buildPrompt(question, responseType, context)
        val answer = generate(prompt)
        return QueryResult(answer = answer, context = context)
    }

    suspend fun buildContext(question: String): List<QueryContextChunk> {
        val queryEmbedding = embed(question) ?: return emptyList()
        val scored =
            communityReports
                .mapNotNull { report ->
                    val summaryEmbedding = embed(report.summary) ?: return@mapNotNull null
                    val score = cosineSimilarity(queryEmbedding, summaryEmbedding)
                    QueryContextChunk(id = report.communityId.toString(), text = report.summary, score = score)
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
                "- [community ${chunk.id}] ${chunk.text.take(maxContextChars)}"
            }
        return """
            You are a helpful assistant answering using high-level community reports from a knowledge graph.
            Provide the response in the form: $responseType.

            Community reports:
            $contextBlock

            Question: $question

            Answer:
            """.trimIndent()
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

    private val responder: ContextResponder =
        AiServices.create(ContextResponder::class.java, chatModel)
}

/**
 * DRIFT search is a hybrid: it gathers global (community) context and combines
 * it with local entity/text snippets before asking the model to respond.
 */
class DriftQueryEngine(
    private val chatModel: OpenAiChatModel,
    private val globalEngine: GlobalQueryEngine,
    private val localEngine: LocalQueryEngine,
    private val maxCombinedContext: Int = 8,
) {
    suspend fun answer(
        question: String,
        responseType: String,
    ): QueryResult {
        val globalContext = globalEngine.buildContext(question)
        val localContext = localEngine.buildContext(question)
        val combined =
            (globalContext + localContext)
                .distinctBy { it.id }
                .sortedByDescending { it.score }
                .take(maxCombinedContext)

        val prompt = buildPrompt(question, responseType, globalContext, localContext)
        val answer = generate(prompt)
        return QueryResult(answer = answer, context = combined)
    }

    private fun buildPrompt(
        question: String,
        responseType: String,
        globalContext: List<QueryContextChunk>,
        localContext: List<QueryContextChunk>,
    ): String {
        val globalBlock =
            if (globalContext.isEmpty()) {
                "No community reports were found."
            } else {
                globalContext.joinToString("\n\n") { "- [community ${it.id}] ${it.text.take(800)}" }
            }
        val localBlock =
            if (localContext.isEmpty()) {
                "No local entity context was found."
            } else {
                localContext.joinToString("\n\n") { "- [${it.id}] ${it.text.take(800)}" }
            }
        return """
            You are a helpful assistant using both global (community) and local (entity/text) context to answer the question.
            Provide the response in the form: $responseType.

            Global context:
            $globalBlock

            Local context:
            $localBlock

            Question: $question

            Answer:
            """.trimIndent()
    }

    private suspend fun generate(prompt: String): String =
        withContext(Dispatchers.IO) {
            runCatching { responder.answer(prompt) }.getOrNull() ?: "No response generated."
        }

    private val responder: ContextResponder =
        AiServices.create(ContextResponder::class.java, chatModel)
}

private interface ContextResponder {
    @SystemMessage("You are a helpful assistant. Answer the question using only the provided context.")
    fun answer(
        @UserMessage prompt: String,
    ): String
}
