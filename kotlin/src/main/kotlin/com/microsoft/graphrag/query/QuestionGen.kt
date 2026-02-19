package com.microsoft.graphrag.query

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingType
import com.microsoft.graphrag.prompts.query.QUESTION_SYSTEM_PROMPT
import com.microsoft.graphrag.query.LocalSearchContextBuilder.ConversationHistory
import com.microsoft.graphrag.query.LocalSearchContextBuilder.ConversationTurn
import dev.langchain4j.data.message.SystemMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.chat.response.ChatResponse
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler
import dev.langchain4j.model.openai.OpenAiStreamingChatModel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.future.await
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonPrimitive
import java.util.concurrent.CompletableFuture

/**
 * Result of a question generation request.
 *
 * @property response Generated questions returned by the model.
 * @property contextData Context records used to generate the questions.
 * @property completionTime Total generation time in seconds.
 * @property llmCalls Number of LLM calls performed.
 * @property promptTokens Token count of the prompt used.
 */
data class QuestionResult(
    val response: List<String>,
    val contextData: Map<String, List<Map<String, String>>>,
    val completionTime: Double,
    val llmCalls: Int,
    val promptTokens: Int,
)

/**
 * Local question generator that builds context and streams responses from a chat model.
 *
 * @property model Streaming chat model used for generation.
 * @property contextBuilder Builder used to assemble local context.
 * @property callbacks Optional callbacks to receive context records.
 * @property encoding Token encoder used for prompt accounting.
 */
class LocalQuestionGen(
    private val model: OpenAiStreamingChatModel,
    private val contextBuilder: LocalSearchContextBuilder,
    private val callbacks: List<QueryCallbacks> = emptyList(),
    private val encoding: Encoding = Encodings.newLazyEncodingRegistry().getEncoding(EncodingType.CL100K_BASE),
) {
    /**
     * Generate one or more questions using the streaming chat model, optionally building or using provided
     * contextual records.
     *
     * If `questionHistory` is non-empty, the last element is treated as the current question and earlier
     * elements are used as prior conversation turns. If `contextData` is null, this function invokes the
     * context builder with the provided ranking and top-K parameters to assemble context; otherwise it uses the
     * supplied `contextData` as a single in-context record. The generated questions are produced by formatting
     * a system prompt with the assembled context and invoking the streaming model.
     *
     * @param questionHistory List of prior user questions; the last item is the current question and earlier
     * items become conversation history.
     * @param contextData Optional raw context text to use instead of building context.
     * @param questionCount Number of questions to request from the model.
     * @param conversationHistoryMaxTurns Maximum number of prior turns to include in conversation history when
     * building context.
     * @param maxContextTokens Maximum tokens allowed for assembled context.
     * @param textUnitProp Weight applied to text-unit relevance when building context.
     * @param communityProp Weight applied to community relevance when building context.
     * @param topKMappedEntities Number of top mapped entities to include in context.
     * @param topKRelationships Number of top relationships to include in context.
     * @param topKClaims Number of top claims to include in context.
     * @param topKCommunities Number of top communities to include in context.
     * @param includeCommunityRank Include community rank scores in the built context when true.
     * @param includeEntityRank Include entity rank scores in the built context when true.
     * @param includeRelationshipWeight Include relationship weight values in the built context when true.
     * @return A QuestionResult containing:
     * - `response`: a list of generated question strings (each non-empty line from the model, trimmed of
     * leading "- "),
     *   - `contextData`: the context records used (including a `question_context` entry),
     *   - `completionTime`: total generation time in seconds,
     *   - `llmCalls`: number of LLM calls performed,
     *   - `promptTokens`: token count of the system prompt used.
     */
    @Suppress("LongParameterList")
    suspend fun generate(
        questionHistory: List<String>,
        contextData: String?,
        questionCount: Int,
        conversationHistoryMaxTurns: Int = 5,
        maxContextTokens: Int = 8000,
        textUnitProp: Double = 0.5,
        communityProp: Double = 0.25,
        topKMappedEntities: Int = 10,
        topKRelationships: Int = 10,
        topKClaims: Int = 10,
        topKCommunities: Int = 5,
        includeCommunityRank: Boolean = false,
        includeEntityRank: Boolean = false,
        includeRelationshipWeight: Boolean = false,
    ): QuestionResult {
        val startTime = System.currentTimeMillis()

        val (questionText, conversationHistory) =
            if (questionHistory.isEmpty()) {
                Pair("", null)
            } else {
                val history = questionHistory.dropLast(1).map { ConversationTurn(ConversationTurn.Role.USER, it) }
                Pair(questionHistory.last(), ConversationHistory(history))
            }

        val (finalContextData, contextRecords) =
            if (contextData == null) {
                val result =
                    contextBuilder.buildContext(
                        query = questionText,
                        conversationHistory = conversationHistory,
                        conversationHistoryMaxTurns = conversationHistoryMaxTurns,
                        maxContextTokens = maxContextTokens,
                        textUnitProp = textUnitProp,
                        communityProp = communityProp,
                        topKMappedEntities = topKMappedEntities,
                        topKRelationships = topKRelationships,
                        topKClaims = topKClaims,
                        topKCommunities = topKCommunities,
                        includeCommunityRank = includeCommunityRank,
                        includeEntityRank = includeEntityRank,
                        includeRelationshipWeight = includeRelationshipWeight,
                        returnCandidateContext = true,
                        contextCallbacks = callbacks.map { cb -> { res -> cb.onContext(res.contextRecords.toMutableContextRecords()) } },
                    )

                Pair(result.contextText, result.contextRecords)
            } else {
                val records = mutableMapOf<String, MutableList<MutableMap<String, String>>>()
                records["context_data"] = mutableListOf(mutableMapOf("text" to contextData, "in_context" to "true"))
                Pair(contextData, records)
            }

        val systemPrompt =
            formatPrompt(
                contextData = finalContextData,
                questionCount = questionCount,
            )

        val response = streamingChat(systemPrompt, questionText)

        val completionTime = (System.currentTimeMillis() - startTime) / 1000.0

        return QuestionResult(
            response = parseQuestions(response),
            contextData =
                contextRecords
                    .plus(
                        mapOf(
                            "question_context" to
                                mutableListOf(mutableMapOf("text" to questionText, "in_context" to "true")),
                        ),
                    ).toImmutableContextRecords(),
            completionTime = completionTime,
            llmCalls = 1,
            promptTokens = encoding.countTokens(systemPrompt),
        )
    }

    @Suppress("ReturnCount")
    private fun parseQuestions(raw: String): List<String> {
        val fallback =
            raw
                .split("\n")
                .map { it.removePrefix("- ").trim() }
                .filter { it.isNotBlank() }
        return runCatching {
            val element = Json.parseToJsonElement(raw)
            val obj = element as? JsonObject ?: return fallback
            val questions = obj["questions"] as? JsonArray ?: return fallback
            questions.mapNotNull { it.jsonPrimitive.contentOrNull?.trim() }.filter { it.isNotBlank() }
        }.getOrElse { fallback }
    }

    /**
     * Streams generated question text from the LLM, using either provided context or context built from the
     * conversation history.
     *
     * Builds or reuses context, notifies callbacks with context updates and new tokens, and emits partial
     * response chunks as they arrive from the model.
     *
     * @param questionHistory List of prior question strings; the last element is treated as the current
     * question and earlier elements become conversation turns.
     * @param contextData Optional precomputed context text; when provided, it is used directly instead of
     * invoking the context builder.
     * @param questionCount Number of questions to request or include in the prompt template.
     * @param conversationHistoryMaxTurns Maximum number of previous turns to include when constructing
     * conversation history for context building.
     * @param maxContextTokens Maximum number of tokens allowed for built context.
     * @param textUnitProp Weight (0.0-1.0) controlling the influence of text-unit relevance when building context.
     * @param communityProp Weight (0.0-1.0) controlling the influence of community relevance when building context.
     * @param topKMappedEntities Number of top mapped entities to include in built context.
     * @param topKRelationships Number of top relationships to include in built context.
     * @param topKClaims Number of top claims to include in built context.
     * @param topKCommunities Number of top communities to include in built context.
     * @param includeCommunityRank Whether to include community ranking information in the built context.
     * @param includeEntityRank Whether to include entity ranking information in the built context.
     * @param includeRelationshipWeight Whether to include relationship weight information in the built context.
     * @return A Flow that emits partial response strings (token chunks) from the streaming LLM as they are produced.
     */
    @Suppress("LongParameterList")
    fun streamGenerate(
        questionHistory: List<String>,
        contextData: String?,
        questionCount: Int,
        conversationHistoryMaxTurns: Int = 5,
        maxContextTokens: Int = 8000,
        textUnitProp: Double = 0.5,
        communityProp: Double = 0.25,
        topKMappedEntities: Int = 10,
        topKRelationships: Int = 10,
        topKClaims: Int = 10,
        topKCommunities: Int = 5,
        includeCommunityRank: Boolean = false,
        includeEntityRank: Boolean = false,
        includeRelationshipWeight: Boolean = false,
    ): Flow<String> =
        callbackFlow {
            val (questionText, conversationHistory) =
                if (questionHistory.isEmpty()) {
                    Pair("", null)
                } else {
                    val history = questionHistory.dropLast(1).map { ConversationTurn(ConversationTurn.Role.USER, it) }
                    Pair(questionHistory.last(), ConversationHistory(history))
                }

            val (finalContextData, contextRecords) =
                if (contextData == null) {
                    val result =
                        contextBuilder.buildContext(
                            query = questionText,
                            conversationHistory = conversationHistory,
                            conversationHistoryMaxTurns = conversationHistoryMaxTurns,
                            maxContextTokens = maxContextTokens,
                            textUnitProp = textUnitProp,
                            communityProp = communityProp,
                            topKMappedEntities = topKMappedEntities,
                            topKRelationships = topKRelationships,
                            topKClaims = topKClaims,
                            topKCommunities = topKCommunities,
                            includeCommunityRank = includeCommunityRank,
                            includeEntityRank = includeEntityRank,
                            includeRelationshipWeight = includeRelationshipWeight,
                            returnCandidateContext = true,
                            contextCallbacks =
                                callbacks.map { cb ->
                                    { res -> cb.onContext(res.contextRecords.toMutableContextRecords()) }
                                },
                        )
                    val mutableRecords =
                        result.contextRecords
                            .mapValues { entry -> entry.value.toMutableList() }
                            .toMutableMap()
                    Pair(result.contextText, mutableRecords)
                } else {
                    val records = mutableMapOf<String, MutableList<MutableMap<String, String>>>()
                    records["context_data"] = mutableListOf(mutableMapOf("text" to contextData, "in_context" to "true"))
                    Pair(contextData, records)
                }

            contextRecords["question_context"] =
                mutableListOf(mutableMapOf("text" to questionText, "in_context" to "true"))
            callbacks.forEach { it.onContext(contextRecords) }

            val systemPrompt =
                formatPrompt(
                    contextData = finalContextData,
                    questionCount = questionCount,
                )

            model.chat(
                listOf(
                    SystemMessage(systemPrompt),
                    UserMessage(questionText),
                ),
                object : StreamingChatResponseHandler {
                    override fun onPartialResponse(partialResponse: String) {
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
     * Collects partial responses from the streaming chat model and concatenates them into a single string.
     *
     * @param prompt The full system prompt sent to the model (including injected context and question).
     * @return The complete concatenated response produced by the model.
     */
    private suspend fun streamingChat(
        systemPrompt: String,
        userMessage: String,
    ): String {
        val future = CompletableFuture<String>()

        val responseBuilder = StringBuilder()

        model.chat(
            listOf(
                SystemMessage(systemPrompt),
                UserMessage(userMessage),
            ),
            object : StreamingChatResponseHandler {
                override fun onPartialResponse(partialResponse: String) {
                    responseBuilder.append(partialResponse)
                    callbacks.forEach { it.onLLMNewToken(partialResponse) }
                }

                override fun onCompleteResponse(response: ChatResponse) {
                    future.complete(responseBuilder.toString())
                }

                override fun onError(error: Throwable) {
                    future.completeExceptionally(error)
                }
            },
        )

        return future.await()
    }

    /**
     * Builds the system prompt by replacing template placeholders with the provided context and question count.
     *
     * @param contextData Text to substitute for the `{context_data}` placeholder in the system prompt.
     * @param questionCount Number to substitute for the `{question_count}` placeholder in the system prompt.
     * @return The finalized system prompt string with placeholders replaced.
     */
    private fun formatPrompt(
        contextData: String,
        questionCount: Int,
    ): String =
        QUESTION_SYSTEM_PROMPT
            .replace("{context_data}", contextData)
            .replace("{question_count}", questionCount.toString())
}
