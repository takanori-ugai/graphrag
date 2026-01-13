


package com.microsoft.graphrag.query

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingType
import com.microsoft.graphrag.prompts.query.QUESTION_SYSTEM_PROMPT
import com.microsoft.graphrag.query.LocalSearchContextBuilder.ConversationHistory
import com.microsoft.graphrag.query.LocalSearchContextBuilder.ConversationTurn
import dev.langchain4j.model.chat.response.ChatResponse
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler
import dev.langchain4j.model.openai.OpenAiStreamingChatModel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import java.util.concurrent.CompletableFuture

data class QuestionResult(
    val response: List<String>,
    val contextData: Map<String, List<MutableMap<String, String>>>,
    val completionTime: Double,
    val llmCalls: Int,
    val promptTokens: Int,
)

class LocalQuestionGen(
    private val model: OpenAiStreamingChatModel,
    private val contextBuilder: LocalSearchContextBuilder,
    private val callbacks: List<QueryCallbacks> = emptyList(),
    private val encoding: Encoding = Encodings.newLazyEncodingRegistry().getEncoding(EncodingType.CL100K_BASE),
) {
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
                        contextCallbacks = callbacks.map { cb -> { res -> cb.onContext(res.contextRecords) } },
                    )

                Pair(result.contextText, result.contextRecords)
            } else {
                val records = mutableMapOf<String, MutableList<MutableMap<String, String>>>()
                records["context_data"] = mutableListOf(mutableMapOf("text" to contextData, "in_context" to "true"))
                Pair(contextData, records)
            }

        val systemPrompt =
            QUESTION_SYSTEM_PROMPT
                .replace("{context_data}", finalContextData)
                .replace("{question_count}", questionCount.toString())

        val response = streamingChat("$systemPrompt\n\nUser question: $questionText")

        val completionTime = (System.currentTimeMillis() - startTime) / 1000.0

        return QuestionResult(
            response = response.split("\n").map { it.removePrefix("- ").trim() }.filter { it.isNotBlank() },
            contextData =
                contextRecords +
                    mapOf("question_context" to mutableListOf(mutableMapOf("text" to questionText, "in_context" to "true"))),
            completionTime = completionTime,
            llmCalls = 1,
            promptTokens = encoding.countTokens(systemPrompt),
        )
    }

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
                            contextCallbacks = callbacks.map { cb -> { res -> cb.onContext(res.contextRecords) } },
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
                QUESTION_SYSTEM_PROMPT
                    .replace("{context_data}", finalContextData)
                    .replace("{question_count}", questionCount.toString())
            val fullPrompt = "$systemPrompt\n\nUser question: $questionText"

            val responseBuilder = StringBuilder()
            model.chat(
                fullPrompt,
                object : StreamingChatResponseHandler {
                    override fun onPartialResponse(partialResponse: String) {
                        responseBuilder.append(partialResponse)
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

    private suspend fun streamingChat(prompt: String): String {
        val future = CompletableFuture<String>()

        val responseBuilder = StringBuilder()

        model.chat(
            prompt,
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

        return future.get()
    }
}
