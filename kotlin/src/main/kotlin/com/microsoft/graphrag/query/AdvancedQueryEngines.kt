package com.microsoft.graphrag.query

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingType
import com.microsoft.graphrag.index.Claim
import com.microsoft.graphrag.index.CommunityAssignment
import com.microsoft.graphrag.index.CommunityReport
import com.microsoft.graphrag.index.CommunityReportEmbedding
import com.microsoft.graphrag.index.Covariate
import com.microsoft.graphrag.index.Entity
import com.microsoft.graphrag.index.EntitySummary
import com.microsoft.graphrag.index.LocalVectorStore
import com.microsoft.graphrag.index.Relationship
import com.microsoft.graphrag.index.TextEmbedding
import com.microsoft.graphrag.index.TextUnit
import com.microsoft.graphrag.query.LocalSearchContextBuilder.ConversationHistory
import com.microsoft.graphrag.query.QueryCallbacks
import dev.langchain4j.model.chat.response.ChatResponse
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.model.openai.OpenAiStreamingChatModel
import dev.langchain4j.model.output.Response
import dev.langchain4j.service.AiServices
import dev.langchain4j.service.SystemMessage
import dev.langchain4j.service.UserMessage
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.future.await
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonPrimitive
import java.util.concurrent.CompletableFuture
import kotlin.math.sqrt

/**
 * Local search mirrors the Python local search by prioritizing entity-centric context (entity summaries if available,
 * otherwise source chunks). It now also blends in relationships, claims, community reports, and optional conversation
 * history, falling back to text-unit similarity if no entity vectors are present.
 */
@Suppress("LongParameterList")
class LocalQueryEngine(
    private val streamingModel: OpenAiStreamingChatModel,
    private val embeddingModel: EmbeddingModel,
    private val vectorStore: LocalVectorStore,
    private val textUnits: List<TextUnit>,
    private val textEmbeddings: List<TextEmbedding>,
    private val entities: List<Entity>,
    private val entitySummaries: List<EntitySummary>,
    private val relationships: List<Relationship>,
    private val claims: List<Claim>,
    private val covariates: Map<String, List<Covariate>>,
    private val communities: List<CommunityAssignment>,
    private val communityReports: List<CommunityReport>,
    private val topKEntities: Int = 5,
    private val topKRelationships: Int = 10,
    private val topKClaims: Int = 10,
    private val topKCommunities: Int = 3,
    private val maxContextTokens: Int = 12_000,
    private val columnDelimiter: String = "|",
    private val maxContextChars: Int = 800,
    private val callbacks: List<QueryCallbacks> = emptyList(),
    private val modelParams: ModelParams = ModelParams(jsonResponse = true),
    private val systemPrompt: String = DEFAULT_LOCAL_SEARCH_SYSTEM_PROMPT,
    private val textUnitProp: Double = 0.5,
    private val communityProp: Double = 0.25,
    private val conversationHistoryMaxTurns: Int = 5,
    private val encoding: Encoding = Encodings.newLazyEncodingRegistry().getEncoding(EncodingType.CL100K_BASE),
) {
    /**
     * Generates an answer to the given question by building a local context, invoking the model, and aggregating results.
     *
     * Builds context from entities, relationships, claims, communities, and optional conversation history, constructs a prompt
     * (optionally requesting JSON), runs the model to produce an answer, and aggregates context, follow-up queries, a score,
     * and LLM/token usage metrics into a QueryResult.
     *
     * @param question The user's question to answer.
     * @param responseType A descriptor that influences the style or format of the response (injected into the prompt).
     * @param conversationHistory Optional recent conversation turns to include in context; empty list means no history.
     * @param driftQuery Optional drift/global query string to modify or augment the prompt/context; null to omit.
     * @return A QueryResult containing the model's answer text, the context chunks and context records used, any follow-up
     * queries, an optional score, and aggregated LLM call and token usage metrics.
     */
    suspend fun answer(
        question: String,
        responseType: String,
        conversationHistory: List<String> = emptyList(),
        driftQuery: String? = null,
    ): QueryResult {
        val contextResult =
            contextBuilder.buildContext(
                query = question,
                conversationHistory = toConversationHistory(conversationHistory),
                conversationHistoryMaxTurns = conversationHistoryMaxTurns,
                maxContextTokens = maxContextTokens,
                topKMappedEntities = topKEntities,
                topKRelationships = topKRelationships,
                topKClaims = topKClaims,
                topKCommunities = topKCommunities,
                textUnitProp = textUnitProp,
                communityProp = communityProp,
                includeCommunityRank = true,
                includeEntityRank = true,
                includeRelationshipWeight = true,
                returnCandidateContext = true,
                contextCallbacks = callbacks.map { cb -> { res -> cb.onContext(res.contextRecords.toMutableContextRecords()) } },
            )
        val promptBase = buildPrompt(responseType, contextResult.contextText, driftQuery)
        val promptWithJson =
            if (modelParams.jsonResponse) {
                "$promptBase\n\nReturn ONLY valid JSON per the schema above."
            } else {
                promptBase
            }
        val finalPrompt = "$promptWithJson\n\nUser question: $question"
        callbacks.forEach { it.onContext(contextResult.contextRecords) }
        val parsed = generate(finalPrompt)
        val promptTokens = encoding.countTokens(finalPrompt)
        val answerTokens = encoding.countTokens(parsed.answer)
        val llmCallsCategories = mapOf("response" to 1, "build_context" to contextResult.llmCalls)
        val promptTokensCategories = mapOf("response" to promptTokens, "build_context" to contextResult.promptTokens)
        val outputTokensCategories = mapOf("response" to answerTokens, "build_context" to contextResult.outputTokens)
        return QueryResult(
            answer = parsed.answer,
            context = contextResult.contextChunks,
            contextRecords = contextResult.contextRecords.toImmutableContextRecords(),
            followUpQueries = parsed.followUps,
            score = parsed.score,
            llmCalls = contextResult.llmCalls + 1,
            promptTokens = contextResult.promptTokens + promptTokens,
            outputTokens = contextResult.outputTokens + answerTokens,
            llmCallsCategories = llmCallsCategories,
            promptTokensCategories = promptTokensCategories,
            outputTokensCategories = outputTokensCategories,
        )
    }

    /**
     * Produces context chunks relevant to the supplied question and optional conversation history.
     *
     * @param question The user's query to find contextual chunks for.
     * @param conversationHistory Optional prior messages in chronological order to provide conversational context; pass an empty list for no history.
     * @return A list of QueryContextChunk containing the selected and ranked context pieces for the query.
     */
    suspend fun buildContext(
        question: String,
        conversationHistory: List<String> = emptyList(),
    ): List<QueryContextChunk> =
        contextBuilder
            .buildContext(
                query = question,
                conversationHistory = toConversationHistory(conversationHistory),
                conversationHistoryMaxTurns = conversationHistoryMaxTurns,
                maxContextTokens = maxContextTokens,
                topKMappedEntities = topKEntities,
                topKRelationships = topKRelationships,
                topKClaims = topKClaims,
                topKCommunities = topKCommunities,
                textUnitProp = textUnitProp,
                communityProp = communityProp,
            ).contextChunks

    /**
     * Builds the system prompt by injecting context, the response type, and an optional global query.
     *
     * Replaces `{context_data}` with `context` and `{response_type}` with `responseType`. If the
     * resulting prompt contains `{global_query}` that placeholder is replaced with `driftQuery`
     * (or an empty string when `driftQuery` is null). If there is no `{global_query}` placeholder
     * and `driftQuery` is non-blank, the function appends a new line with `Global query: <driftQuery>`.
     *
     * @param responseType A short descriptor of the expected response format or role to inject.
     * @param context The assembled context block to include in the prompt.
     * @param driftQuery An optional global query to either replace `{global_query}` or be appended when present.
     * @return The final prompt string with all applicable replacements and optional appended global query.
     */
    private fun buildPrompt(
        responseType: String,
        context: String,
        driftQuery: String?,
    ): String =
        systemPrompt
            .replace("{context_data}", context)
            .replace("{response_type}", responseType)
            .let { prompt ->
                if (prompt.contains("{global_query}")) {
                    prompt.replace("{global_query}", driftQuery ?: "")
                } else if (!driftQuery.isNullOrBlank()) {
                    "$prompt\n\nGlobal query: $driftQuery"
                } else {
                    prompt
                }
            }

    /**
     * Streams a completion for the given prompt, aggregates the emitted tokens, and parses the final response into a ParsedAnswer.
     *
     * Streams tokens from the configured streaming model, forwarding each partial token to registered callbacks. If the model produces a full response it is parsed as structured JSON (if possible) into a ParsedAnswer; on generation failure the raw text "No response generated." is parsed and returned.
     *
     * @param prompt The full prompt to send to the streaming model.
     * @return A ParsedAnswer representing the final parsed response (including any follow-up queries or score if present).
     */
    private suspend fun generate(prompt: String): ParsedAnswer {
        val builder = StringBuilder()
        val future = CompletableFuture<String>()
        streamingModel.chat(
            prompt,
            object : StreamingChatResponseHandler {
                override fun onPartialResponse(partialResponse: String) {
                    builder.append(partialResponse)
                    callbacks.forEach { it.onLLMNewToken(partialResponse) }
                }

                override fun onCompleteResponse(response: ChatResponse) {
                    future.complete(builder.toString())
                }

                override fun onError(error: Throwable) {
                    future.completeExceptionally(error)
                }
            },
        )
        val raw = runCatching { future.await() }.getOrElse { "No response generated." }
        return parseStructuredAnswer(raw)
    }

    /**
     * Streams model-generated tokens for a question using a locally built context and notifies callbacks as tokens arrive.
     *
     * Builds a context from local data and optional conversation history, constructs the final prompt (optionally enforcing a JSON-only response), and emits partial response tokens from the streaming model as they are produced.
     *
     * @param question The user's question to answer.
     * @param responseType The desired response type or format hint inserted into the prompt.
     * @param conversationHistory Past user messages in chronological order; empty by default.
     * @param driftQuery Optional global/drift query to include or replace context-related placeholders in the prompt.
     * @return A Flow that emits partial response tokens (`String`) as the streaming model produces them; the flow completes when the model finishes or fails with an error.
     */
    fun streamAnswer(
        question: String,
        responseType: String,
        conversationHistory: List<String> = emptyList(),
        driftQuery: String? = null,
    ): Flow<String> =
        callbackFlow {
            val contextResult =
                contextBuilder.buildContext(
                    query = question,
                    conversationHistory = toConversationHistory(conversationHistory),
                    conversationHistoryMaxTurns = conversationHistoryMaxTurns,
                    maxContextTokens = maxContextTokens,
                    topKMappedEntities = topKEntities,
                    topKRelationships = topKRelationships,
                    topKClaims = topKClaims,
                    topKCommunities = topKCommunities,
                    textUnitProp = textUnitProp,
                    communityProp = communityProp,
                    includeCommunityRank = true,
                    includeEntityRank = true,
                    includeRelationshipWeight = true,
                    returnCandidateContext = true,
                    contextCallbacks = callbacks.map { cb -> { res -> cb.onContext(res.contextRecords.toMutableContextRecords()) } },
                )
            val promptBase = buildPrompt(responseType, contextResult.contextText, driftQuery)
            val promptWithJson =
                if (modelParams.jsonResponse) {
                    "$promptBase\n\nReturn ONLY valid JSON per the schema above."
                } else {
                    promptBase
                }
            callbacks.forEach { it.onContext(contextResult.contextRecords) }
            val finalPrompt = "$promptWithJson\n\nUser question: $question"
            val builder = StringBuilder()
            streamingModel.chat(
                finalPrompt,
                object : StreamingChatResponseHandler {
                    override fun onPartialResponse(partialResponse: String) {
                        builder.append(partialResponse)
                        callbacks.forEach { it.onLLMNewToken(partialResponse) }
                        trySend(partialResponse)
                    }

                    override fun onCompleteResponse(response: ChatResponse) {
                        close()
                    }

                    override fun onError(error: Throwable) {
                        close(error)
                    }
                },
            )
            awaitClose {}
        }

    /**
     * Parses a raw model response string into a ParsedAnswer containing an answer, follow-up queries, and an optional score.
     *
     * If `raw` is a JSON object with a `response` string, an optional `follow_up_queries` array of strings, and an optional numeric `score`,
     * those values are extracted and returned. If parsing fails or the JSON is not an object, returns a fallback ParsedAnswer with
     * the original `raw` as the answer, an empty follow-up list, and a null score.
     *
     * @return A ParsedAnswer with `answer`, `followUps`, and `score` populated from the input when available, or a fallback containing the raw text otherwise.
     */
    private fun parseStructuredAnswer(raw: String): ParsedAnswer {
        val fallback = ParsedAnswer(raw, emptyList(), null)
        return runCatching {
            val element = Json.parseToJsonElement(raw)
            val obj = element as? JsonObject ?: return fallback
            val response = obj["response"]?.jsonPrimitive?.content ?: raw
            val followUps =
                (obj["follow_up_queries"] as? JsonArray)
                    ?.mapNotNull { it.jsonPrimitive.contentOrNull }
                    .orEmpty()
            val score = obj["score"]?.jsonPrimitive?.doubleOrNull
            ParsedAnswer(response, followUps, score)
        }.getOrElse { fallback }
    }

    private data class ParsedAnswer(
        val answer: String,
        val followUps: List<String>,
        val score: Double?,
    )

    private val contextBuilder =
        LocalSearchContextBuilder(
            embeddingModel = embeddingModel,
            vectorStore = vectorStore,
            textUnits = textUnits,
            textEmbeddings = textEmbeddings,
            entities = entities,
            entitySummaries = entitySummaries,
            relationships = relationships,
            claims = claims,
            covariates = covariates,
            communities = communities,
            communityReports = communityReports,
            columnDelimiter = columnDelimiter,
        )

    /**
     * Create a ConversationHistory from a list of user messages.
     *
     * @param history The list of user messages in chronological order.
     * @return A `ConversationHistory` with each message wrapped as a user `ConversationTurn`, or `null` if `history` is empty.
     */
    private fun toConversationHistory(history: List<String>): LocalSearchContextBuilder.ConversationHistory? =
        if (history.isEmpty()) {
            null
        } else {
            LocalSearchContextBuilder.ConversationHistory(
                history.map { LocalSearchContextBuilder.ConversationTurn(LocalSearchContextBuilder.ConversationTurn.Role.USER, it) },
            )
        }
}

/**
 * Global search leans on community reports; it embeds the summaries and selects
 * the most relevant communities as context.
 */
class GlobalQueryEngine(
    private val chatModel: OpenAiChatModel,
    private val embeddingModel: EmbeddingModel,
    private val communityReports: List<CommunityReport>,
    private val communityReportEmbeddings: List<CommunityReportEmbedding> = emptyList(),
    private val topK: Int = 3,
    private val maxContextChars: Int = 1000,
) {
    private val logger = KotlinLogging.logger {}
    private val communityEmbeddingCache =
        mutableMapOf<Int, List<Double>>().apply {
            communityReportEmbeddings.forEach { put(it.communityId, it.vector) }
        }

    /**
     * Produce an answer for the given question formatted according to the specified response type.
     *
     * @param question The user's question to be answered.
     * @param responseType A descriptor that controls the format or style of the response (for example, a directive like `"short_answer"` or `"json"`).
     * @return A `QueryResult` containing the generated answer and the context chunks that were used to produce it.
     */
    suspend fun answer(
        question: String,
        responseType: String,
    ): QueryResult {
        val context = buildContext(question)
        val prompt = buildPrompt(responseType, context)
        val answer = generate(prompt)
        return QueryResult(answer = answer, context = context)
    }

    /**
     * Builds a ranked list of community-level context chunks relevant to the provided question.
     *
     * Embeds the question, compares it to cached or newly computed embeddings of each community report
     * using cosine similarity, caches any newly computed report embeddings, excludes reports whose
     * embeddings could not be obtained, and returns the top-k results sorted by descending score.
     *
     * @param question The user query to use when scoring community report relevance.
     * @return A list of `QueryContextChunk` objects (id, text, score) representing the most relevant
     * communities, sorted by descending relevance score.
     */
    suspend fun buildContext(question: String): List<QueryContextChunk> {
        val queryEmbedding = embed(question) ?: return emptyList()
        val scored =
            communityReports
                .mapNotNull { report ->
                    val summaryEmbedding =
                        communityEmbeddingCache[report.communityId]
                            ?: embed(report.summary)?.also { communityEmbeddingCache[report.communityId] = it }
                            ?: return@mapNotNull null
                    val score = cosineSimilarity(queryEmbedding, summaryEmbedding)
                    QueryContextChunk(id = report.communityId.toString(), text = report.summary, score = score)
                }.sortedByDescending { it.score }
                .take(topK)
        return scored
    }

    /**
     * Builds the system prompt by injecting the provided report context and response type into the reduce-system template.
     *
     * @param responseType A short descriptor of the desired response style or intent that will replace `{response_type}` in the template.
     * @param context A list of QueryContextChunk whose `id` and `text` are formatted as `report_id|summary` rows and inserted into the `{report_data}` placeholder.
     * @return The completed system prompt string with the report context block, the specified response type, and the `{max_length}` placeholder set to 500.
     */
    private fun buildPrompt(
        responseType: String,
        context: List<QueryContextChunk>,
    ): String {
        val header = "report_id|summary"
        val rows =
            context.joinToString("\n") { chunk ->
                "${chunk.id}|${chunk.text.take(maxContextChars)}"
            }
        val contextBlock = "$header\n$rows"
        return REDUCE_SYSTEM_PROMPT
            .replace("{report_data}", contextBlock)
            .replace("{response_type}", responseType)
            .replace("{max_length}", "500")
    }

    /**
     * Compute an embedding vector for the given text.
     *
     * @param text The input text to embed.
     * @return The embedding as a list of `Double`, or `null` if embedding could not be obtained.
     */
    private suspend fun embed(text: String): List<Double>? =
        withContext(Dispatchers.IO) {
            runCatching {
                val response: Response<dev.langchain4j.data.embedding.Embedding> = embeddingModel.embed(text)
                response
                    .content()
                    ?.vector()
                    ?.asList()
                    ?.map { it.toDouble() }
            }.getOrElse { error ->
                logger.warn { "Embedding failed for global search ($error)" }
                null
            }
        }

    /**
     * Obtains a model-generated response for the given prompt.
     *
     * If the responder fails or returns null, returns the literal string "No response generated.".
     *
     * @return The responder's text reply, or `"No response generated."` when no reply is available.
     */
    private suspend fun generate(prompt: String): String =
        withContext(Dispatchers.IO) {
            runCatching { responder.answer(prompt) }.getOrNull() ?: "No response generated."
        }

    /**
     * Computes the cosine similarity between two numeric vectors.
     *
     * @param a First vector (embedding) to compare.
     * @param b Second vector (embedding) to compare.
     * @return The cosine similarity in the range [-1.0, 1.0] when both vectors are non-empty, equal-length, and have non-zero magnitude; returns `0.0` if either vector is empty, lengths differ, or either magnitude is zero.
     */
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
    /**
     * Answers a question by combining global and local contexts, building a prompt, and generating a response.
     *
     * @param question The user question to answer.
     * @param responseType The desired response style or format inserted into the prompt.
     * @return A [QueryResult] containing the generated answer and the combined, deduplicated, scored context chunks used to produce it.
     */
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

        val prompt = buildPrompt(question, responseType, combined)
        val answer = generate(prompt)
        return QueryResult(answer = answer, context = combined)
    }

    /**
     * Builds the system prompt for the drift/local hybrid engine by inserting the question,
     * the response type instruction, and a formatted context block into the DRIFT_LOCAL_SYSTEM_PROMPT.
     *
     * The context block is formatted as a table with header `source_id|text` and one row per
     * `QueryContextChunk` in `context`; each row contains the chunk id and up to 800 characters
     * of the chunk text.
     *
     * @param question The user's query to include in the prompt.
     * @param responseType An instruction or label describing the desired response style or format.
     * @param context The list of context chunks to include as source rows in the prompt.
     * @return The final system prompt string with the question, response type, and formatted context embedded.
     */
    private fun buildPrompt(
        question: String,
        responseType: String,
        context: List<QueryContextChunk>,
    ): String {
        val header = "source_id|text"
        val rows =
            context.joinToString("\n") { chunk ->
                "${chunk.id}|${chunk.text.take(800)}"
            }
        val contextBlock = "$header\n$rows"
        return DRIFT_LOCAL_SYSTEM_PROMPT
            .replace("{context_data}", contextBlock)
            .replace("{response_type}", responseType)
            .replace("{global_query}", question)
    }

    /**
     * Obtains a model-generated response for the given prompt.
     *
     * If the responder fails or returns null, returns the literal string "No response generated.".
     *
     * @return The responder's text reply, or `"No response generated."` when no reply is available.
     */
    private suspend fun generate(prompt: String): String =
        withContext(Dispatchers.IO) {
            runCatching { responder.answer(prompt) }.getOrNull() ?: "No response generated."
        }

    private val responder: ContextResponder =
        AiServices.create(ContextResponder::class.java, chatModel)
}

private interface ContextResponder {
    /**
     * Produces an answer to the provided user prompt using only the context contained in that prompt.
     *
     * @param prompt The user message containing the question and any context the assistant should use.
     * @return The assistant's answer as a plain string.
     */
    @SystemMessage("You are a helpful assistant. Answer the question using only the provided context.")
    fun answer(
        @UserMessage prompt: String,
    ): String
}

internal val DEFAULT_LOCAL_SEARCH_SYSTEM_PROMPT =
    """
---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Sources (15, 16), Reports (1), Entities (5, 7); Relationships (23); Claims (2, 7, 34, 46, 64, +more)]."

where 15, 16, 1, 5, 7, 23, 2, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}


---Data tables---

{context_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Sources (15, 16), Reports (1), Entities (5, 7); Relationships (23); Claims (2, 7, 34, 46, 64, +more)]."

where 15, 16, 1, 5, 7, 23, 2, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
    """.trimIndent()

private val REDUCE_SYSTEM_PROMPT =
    """
---Role---

You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

The response should also preserve all the data references previously included in the analysts' reports, but do not mention the roles of multiple analysts in the analysis process.

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.

Limit your response length to {max_length} words.

---Target response length and format---

{response_type}


---Analyst Reports---

{report_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

The response should also preserve all the data references previously included in the analysts' reports, but do not mention the roles of multiple analysts in the analysis process.

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.

Limit your response length to {max_length} words.

---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
    """.trimIndent()

private val DRIFT_LOCAL_SYSTEM_PROMPT =
    """
---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Sources (15, 16)]."

where 15, 16, 1, 5, 7, 23, 2, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Pay close attention specifically to the Sources tables as it contains the most relevant information for the user query. You will be rewarded for preserving the context of the sources in your response.

---Target response length and format---

{response_type}


---Data tables---

{context_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Sources (15, 16)]."

where 15, 16, 1, 5, 7, 23, 2, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Pay close attention specifically to the Sources tables as it contains the most relevant information for the user query. You will be rewarded for preserving the context of the sources in your response.

---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format.

Additionally provide a score between 0 and 100 representing how well the response addresses the overall research question: {global_query}. Based on your response, suggest up to five follow-up questions that could be asked to further explore the topic as it relates to the overall research question. Do not include scores or follow up questions in the 'response' field of the JSON, add them to the respective 'score' and 'follow_up_queries' keys of the JSON output. Format your response in JSON with the following keys and values:

{{'response': str, Put your answer, formatted in markdown, here. Do not answer the global query in this section.
'score': int,
'follow_up_queries': List<String>}}
    """.trimIndent()
