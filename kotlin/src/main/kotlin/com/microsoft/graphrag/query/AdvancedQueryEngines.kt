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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
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
    private val encoding: Encoding = Encodings.newLazyEncodingRegistry().getEncoding(EncodingType.CL100K_BASE),
) {
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
                maxContextTokens = maxContextTokens,
                topKMappedEntities = topKEntities,
                topKRelationships = topKRelationships,
                topKClaims = topKClaims,
                topKCommunities = topKCommunities,
                textUnitProp = 0.5,
                communityProp = 0.25,
                includeCommunityRank = true,
                includeEntityRank = true,
                includeRelationshipWeight = true,
                returnCandidateContext = true,
                contextCallbacks = callbacks.map { cb -> { res -> cb.onContext(res.contextRecords) } },
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
            contextRecords = contextResult.contextRecords,
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

    suspend fun buildContext(
        question: String,
        conversationHistory: List<String> = emptyList(),
    ): List<QueryContextChunk> =
        contextBuilder
            .buildContext(
                query = question,
                conversationHistory = toConversationHistory(conversationHistory),
                maxContextTokens = maxContextTokens,
                topKMappedEntities = topKEntities,
                topKRelationships = topKRelationships,
                topKClaims = topKClaims,
                topKCommunities = topKCommunities,
                textUnitProp = 0.5,
                communityProp = 0.25,
            ).contextChunks

    private fun buildPrompt(
        responseType: String,
        context: String,
        driftQuery: String?,
    ): String =
        LOCAL_SEARCH_SYSTEM_PROMPT
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
        val raw = runCatching { future.get() }.getOrElse { "No response generated." }
        return parseStructuredAnswer(raw)
    }

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
                    maxContextTokens = maxContextTokens,
                    topKMappedEntities = topKEntities,
                    topKRelationships = topKRelationships,
                    topKClaims = topKClaims,
                    topKCommunities = topKCommunities,
                    textUnitProp = 0.5,
                    communityProp = 0.25,
                    includeCommunityRank = true,
                    includeEntityRank = true,
                    includeRelationshipWeight = true,
                    returnCandidateContext = true,
                    contextCallbacks = callbacks.map { cb -> { res -> cb.onContext(res.contextRecords) } },
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
    private val communityEmbeddingCache =
        mutableMapOf<Int, List<Double>>().apply {
            communityReportEmbeddings.forEach { put(it.communityId, it.vector) }
        }

    suspend fun answer(
        question: String,
        responseType: String,
    ): QueryResult {
        val context = buildContext(question)
        val prompt = buildPrompt(responseType, context)
        val answer = generate(prompt)
        return QueryResult(answer = answer, context = context)
    }

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

        val prompt = buildPrompt(question, responseType, combined)
        val answer = generate(prompt)
        return QueryResult(answer = answer, context = combined)
    }

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

private val LOCAL_SEARCH_SYSTEM_PROMPT =
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
