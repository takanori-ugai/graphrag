


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
import java.util.concurrent.CompletableFuture

data class QuestionResult(
    val response: List<String>,
    val contextData: Any,
    val completionTime: Double,
    val llmCalls: Int,
    val promptTokens: Int,
)

class LocalQuestionGen(
    private val model: OpenAiStreamingChatModel,
    private val contextBuilder: LocalSearchContextBuilder,
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
                    )

                Pair(result.contextText, result.contextRecords)
            } else {
                Pair(contextData, mapOf("context_data" to contextData))
            }

        val systemPrompt =
            QUESTION_SYSTEM_PROMPT
                .replace("{context_data}", finalContextData)
                .replace("{question_count}", questionCount.toString())

        val questionMessages = "$systemPrompt\n\nUser question: $questionText"

        val response = streamingChat(questionMessages)

        val completionTime = (System.currentTimeMillis() - startTime) / 1000.0

        return QuestionResult(
            response = response.split("\n").map { it.removePrefix("- ").trim() }.filter { it.isNotBlank() },
            contextData = mapOf("question_context" to questionText) + contextRecords,
            completionTime = completionTime,
            llmCalls = 1,
            promptTokens = encoding.countTokens(systemPrompt),
        )
    }

    private suspend fun streamingChat(prompt: String): String {
        val future = CompletableFuture<String>()

        val responseBuilder = StringBuilder()

        model.chat(
            prompt,
            object : StreamingChatResponseHandler {
                override fun onPartialResponse(partialResponse: String) {
                    responseBuilder.append(partialResponse)
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
