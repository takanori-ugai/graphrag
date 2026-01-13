package com.microsoft.graphrag.query

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingType
import com.microsoft.graphrag.index.LocalVectorStore
import com.microsoft.graphrag.index.TextEmbedding
import com.microsoft.graphrag.index.TextUnit
import dev.langchain4j.model.chat.response.ChatResponse
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.openai.OpenAiStreamingChatModel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import java.util.concurrent.CompletableFuture

data class QueryContextChunk(
    val id: String,
    val text: String,
    val score: Double,
)

data class QueryResult(
    val answer: String,
    val context: List<QueryContextChunk>,
    val contextRecords: Map<String, List<Map<String, String>>> = emptyMap(),
    val contextText: String = "",
    val followUpQueries: List<String> = emptyList(),
    val score: Double? = null,
    val llmCalls: Int = 0,
    val promptTokens: Int = 0,
    val outputTokens: Int = 0,
    val llmCallsCategories: Map<String, Int> = emptyMap(),
    val promptTokensCategories: Map<String, Int> = emptyMap(),
    val outputTokensCategories: Map<String, Int> = emptyMap(),
)

/**
 * Basic search engine that mirrors the Python basic search flow: embed the question, build a token-budgeted CSV context
 * table, and ask the chat model while reporting usage and callback events.
 */
@Suppress("LongParameterList")
class BasicQueryEngine(
    private val streamingModel: OpenAiStreamingChatModel,
    embeddingModel: EmbeddingModel,
    vectorStore: LocalVectorStore,
    textUnits: List<TextUnit>,
    textEmbeddings: List<TextEmbedding>,
    private val topK: Int = 10,
    private val maxContextTokens: Int = 12_000,
    private val columnDelimiter: String = "|",
    private val callbacks: List<QueryCallbacks> = emptyList(),
    private val systemPrompt: String = DEFAULT_BASIC_SEARCH_SYSTEM_PROMPT,
    private val encoding: Encoding = Encodings.newLazyEncodingRegistry().getEncoding(EncodingType.CL100K_BASE),
) {
    suspend fun answer(
        question: String,
        responseType: String,
    ): QueryResult {
        val contextResult =
            contextBuilder.buildContext(
                query = question,
                k = topK,
                maxContextTokens = maxContextTokens,
            )
        callbacks.forEach { it.onContext(contextResult.contextRecords) }
        val prompt = buildPrompt(responseType, contextResult.contextText)
        val promptTokens = encoding.countTokens(prompt)
        val fullPrompt = "$prompt\n\nUser question: $question"
        val answerText = generate(fullPrompt)
        val outputTokens = encoding.countTokens(answerText)

        val llmCallsCategories =
            mapOf(
                "build_context" to contextResult.llmCalls,
                "response" to 1,
            )
        val promptTokensCategories =
            mapOf(
                "build_context" to contextResult.promptTokens,
                "response" to promptTokens,
            )
        val outputTokensCategories =
            mapOf(
                "build_context" to contextResult.outputTokens,
                "response" to outputTokens,
            )

        return QueryResult(
            answer = answerText,
            context = contextResult.contextChunks,
            contextRecords = contextResult.contextRecords.toImmutableContextRecords(),
            contextText = contextResult.contextText,
            llmCalls = llmCallsCategories.values.sum(),
            promptTokens = promptTokensCategories.values.sum(),
            outputTokens = outputTokensCategories.values.sum(),
            llmCallsCategories = llmCallsCategories,
            promptTokensCategories = promptTokensCategories,
            outputTokensCategories = outputTokensCategories,
        )
    }

    fun streamAnswer(
        question: String,
        responseType: String,
    ): Flow<String> =
        callbackFlow {
            val contextResult =
                contextBuilder.buildContext(
                    query = question,
                    k = topK,
                    maxContextTokens = maxContextTokens,
                )
            callbacks.forEach { it.onContext(contextResult.contextRecords) }
            val prompt = buildPrompt(responseType, contextResult.contextText)
            val finalPrompt = "$prompt\n\nUser question: $question"
            streamingModel.chat(
                finalPrompt,
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

    suspend fun buildContext(question: String): List<QueryContextChunk> =
        contextBuilder
            .buildContext(
                query = question,
                k = topK,
                maxContextTokens = maxContextTokens,
            ).contextChunks

    private fun buildPrompt(
        responseType: String,
        context: String,
    ): String =
        systemPrompt
            .replace("{context_data}", context)
            .replace("{response_type}", responseType)

    private suspend fun generate(prompt: String): String {
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
        return runCatching { future.get() }.getOrElse { "No response generated." }
    }

    private val contextBuilder =
        BasicSearchContextBuilder(
            embeddingModel = embeddingModel,
            vectorStore = vectorStore,
            textUnits = textUnits,
            textEmbeddings = textEmbeddings,
            columnDelimiter = columnDelimiter,
            encoding = encoding,
        )
}

internal val DEFAULT_BASIC_SEARCH_SYSTEM_PROMPT =
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


Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
    """.trimIndent()
