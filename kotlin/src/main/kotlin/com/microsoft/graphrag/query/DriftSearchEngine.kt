package com.microsoft.graphrag.query

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingType
import com.microsoft.graphrag.index.CommunityReport
import dev.langchain4j.model.chat.response.ChatResponse
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler
import dev.langchain4j.model.openai.OpenAiStreamingChatModel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonPrimitive
import java.util.concurrent.CompletableFuture

data class DriftSearchResult(
    val answer: String,
    val actions: List<QueryResult>,
    val llmCalls: Int,
    val promptTokens: Int,
    val outputTokens: Int,
    val llmCallsCategories: Map<String, Int>,
    val promptTokensCategories: Map<String, Int>,
    val outputTokensCategories: Map<String, Int>,
)

/**
 * Simplified DRIFT-style search that mirrors the Python DRIFTSearch skeleton:
 * 1) Run a primer global search over community reports.
 * 2) Use the primer's follow-up queries to run local searches.
 * 3) Aggregate answers and usage accounting.
 *
 * NOTE: This is a faithful structural port but uses existing GlobalSearchEngine and LocalQueryEngine.
 */
class DriftSearchEngine(
    private val streamingModel: OpenAiStreamingChatModel,
    private val communityReports: List<CommunityReport>,
    private val globalSearchEngine: GlobalSearchEngine? = null,
    private val localQueryEngine: LocalQueryEngine,
    private val primerSystemPrompt: String = DRIFT_PRIMER_PROMPT,
    private val reduceSystemPrompt: String = DRIFT_REDUCE_PROMPT,
    private val responseType: String = "multiple paragraphs",
    private val callbacks: List<QueryCallbacks> = emptyList(),
    private val primerParams: ModelParams = ModelParams(jsonResponse = true),
    private val reduceParams: ModelParams = ModelParams(jsonResponse = false),
    private val encoding: Encoding = Encodings.newLazyEncodingRegistry().getEncoding(EncodingType.CL100K_BASE),
    private val maxIterations: Int = 3,
) {
    suspend fun search(
        question: String,
        followUpQueries: List<String> = emptyList(),
    ): DriftSearchResult {
        var totalLlmCalls = 0
        var totalPromptTokens = 0
        var totalOutputTokens = 0

        val actions = mutableListOf<QueryResult>()

        // Primer: run DRIFT primer over community reports (fall back to global search if needed)
        val primer =
            runCatching { primerSearch(question) }
                .getOrElse { buildPrimerFallback(question) }
        totalLlmCalls += primer.llmCalls
        totalPromptTokens += primer.promptTokens
        totalOutputTokens += primer.outputTokens
        actions += primer

        // Initialize DRIFT action state
        val state =
            DriftQueryState().apply {
                addActions(if (followUpQueries.isNotEmpty()) followUpQueries else primer.followUpQueries.ifEmpty { listOf(question) })
            }
        var iteration = 0
        while (iteration < maxIterations && state.hasPendingActions()) {
            val nextAction = state.nextPendingAction() ?: break
            val local = localQueryEngine.answer(nextAction.followUpQuery, responseType = responseType)
            totalLlmCalls += local.llmCalls
            totalPromptTokens += local.promptTokens
            totalOutputTokens += local.outputTokens
            nextAction.result = local
            state.completedActions.add(nextAction)
            state.addActions(local.followUpQueries)
            iteration++
        }
        actions.addAll(state.completedActions.mapNotNull { it.result })

        val categories =
            mapOf(
                "primer" to actions.firstOrNull()?.llmCalls.orZero(),
                "local" to actions.drop(1).sumOf { it.llmCalls },
                "reduce" to 1,
            )
        val reduceResult = reduce(question, actions)
        totalLlmCalls += reduceResult.llmCalls
        totalPromptTokens += reduceResult.promptTokens
        totalOutputTokens += reduceResult.outputTokens

        return DriftSearchResult(
            answer = reduceResult.answer.ifBlank { actions.lastOrNull()?.answer ?: "" },
            actions = actions,
            llmCalls = totalLlmCalls,
            promptTokens = totalPromptTokens,
            outputTokens = totalOutputTokens,
            llmCallsCategories = categories,
            promptTokensCategories = mapOf("primer" to totalPromptTokens, "local" to 0, "reduce" to reduceResult.promptTokens),
            outputTokensCategories = mapOf("primer" to totalOutputTokens, "local" to 0, "reduce" to reduceResult.outputTokens),
        )
    }

    fun streamSearch(
        question: String,
        followUpQueries: List<String> = emptyList(),
    ): Flow<String> =
        callbackFlow {
            val primer =
                runBlocking { runCatching { primerSearch(question) }.getOrElse { buildPrimerFallback(question) } }
            trySend(primer.answer)

            val followUps =
                if (followUpQueries.isNotEmpty()) {
                    followUpQueries
                } else {
                    primer.followUpQueries.takeIf { it.isNotEmpty() } ?: listOf(question)
                }

            runBlocking {
                followUps.forEach { followUp ->
                    localQueryEngine
                        .streamAnswer(followUp, responseType = "multiple paragraphs")
                        .collect { partial -> trySend(partial) }
                }
            }
            // Stream reduce
            val contextText =
                (listOf(primer) + runBlocking { followUps.map { localQueryEngine.answer(it, responseType = "multiple paragraphs") } })
                    .mapIndexed { idx, res ->
                        val label = if (idx == 0) "Primer" else "Follow-up $idx"
                        buildString {
                            append("----$label----\n")
                            res.score?.let { append("Score: $it\n") }
                            append(res.answer)
                        }
                    }.joinToString("\n\n")
            val prompt =
                reduceSystemPrompt
                    .replace("{context_data}", contextText)
                    .replace("{response_type}", responseType)
            val fullPrompt = "$prompt\n\nUser question: $question"
            callbacks.forEach { it.onReduceResponseStart(contextText) }
            val reduceBuilder = StringBuilder()
            streamingModel.chat(
                fullPrompt,
                object : StreamingChatResponseHandler {
                    override fun onPartialResponse(partialResponse: String) {
                        reduceBuilder.append(partialResponse)
                        callbacks.forEach { it.onLLMNewToken(partialResponse) }
                        trySend(partialResponse)
                    }

                    override fun onCompleteResponse(response: ChatResponse) {
                        callbacks.forEach { it.onReduceResponseEnd(reduceBuilder.toString()) }
                        close()
                    }

                    override fun onError(error: Throwable) {
                        close(error)
                    }
                },
            )
            awaitClose {}
        }

    private fun Int?.orZero(): Int = this ?: 0

    private fun reduce(
        question: String,
        actions: List<QueryResult>,
    ): QueryResult {
        val contextText =
            actions
                .mapIndexed { idx, res ->
                    val label = if (idx == 0) "Primer" else "Follow-up $idx"
                    buildString {
                        append("----$label----\n")
                        res.score?.let { append("Score: $it\n") }
                        append(res.answer)
                    }
                }.joinToString("\n\n")
        val prompt =
            reduceSystemPrompt
                .replace("{context_data}", contextText)
                .replace("{response_type}", responseType)
                .let { base ->
                    if (reduceParams.jsonResponse) "$base\nReturn ONLY valid JSON per the schema above." else base
                }
        val fullPrompt = "$prompt\n\nUser question: $question"
        val promptTokens = encoding.countTokens(fullPrompt)
        val builder = StringBuilder()
        val future = CompletableFuture<String>()
        callbacks.forEach { it.onReduceResponseStart(contextText) }
        streamingModel.chat(
            fullPrompt,
            object : StreamingChatResponseHandler {
                override fun onPartialResponse(partialResponse: String) {
                    builder.append(partialResponse)
                }

                override fun onCompleteResponse(response: ChatResponse) {
                    future.complete(builder.toString())
                }

                override fun onError(error: Throwable) {
                    future.completeExceptionally(error)
                }
            },
        )
        val answer = runCatching { future.get() }.getOrElse { "" }
        val outputTokens = encoding.countTokens(answer)
        callbacks.forEach { it.onReduceResponseEnd(answer) }
        return QueryResult(
            answer = answer,
            context = emptyList(),
            contextRecords = emptyMap(),
            contextText = contextText,
            llmCalls = 1,
            promptTokens = promptTokens,
            outputTokens = outputTokens,
            llmCallsCategories = mapOf("reduce" to 1),
            promptTokensCategories = mapOf("reduce" to promptTokens),
            outputTokensCategories = mapOf("reduce" to outputTokens),
        )
    }

    private suspend fun primerSearch(question: String): QueryResult {
        val context = buildPrimerContext()
        val prompt =
            primerSystemPrompt
                .replace("{query}", question)
                .replace("{community_reports}", context)
                .let { base ->
                    if (primerParams.jsonResponse) "$base\nReturn ONLY valid JSON per the schema above." else base
                }
        val promptTokens = encoding.countTokens(prompt)
        val builder = StringBuilder()
        val future = CompletableFuture<String>()
        streamingModel.chat(
            prompt,
            object : StreamingChatResponseHandler {
                override fun onPartialResponse(partialResponse: String) {
                    builder.append(partialResponse)
                }

                override fun onCompleteResponse(response: ChatResponse) {
                    future.complete(builder.toString())
                }

                override fun onError(error: Throwable) {
                    future.completeExceptionally(error)
                }
            },
        )
        val raw = runCatching { future.get() }.getOrElse { "" }
        val parsed = parsePrimer(raw, raw)
        val outputTokens = encoding.countTokens(parsed.answer)
        return QueryResult(
            answer = parsed.answer,
            context = emptyList(),
            contextRecords = mapOf("reports" to emptyList()),
            followUpQueries = parsed.followUps,
            score = parsed.score,
            llmCalls = 1,
            promptTokens = promptTokens,
            outputTokens = outputTokens,
            llmCallsCategories = mapOf("primer" to 1),
            promptTokensCategories = mapOf("primer" to promptTokens),
            outputTokensCategories = mapOf("primer" to outputTokens),
        )
    }

    private fun buildPrimerContext(topK: Int = 5): String {
        if (communityReports.isEmpty()) return ""
        return communityReports
            .sortedByDescending { it.rank ?: 0.0 }
            .take(topK)
            .joinToString("\n\n") { report ->
                "Community ${report.communityId} (rank=${report.rank ?: 0.0}): ${report.summary}"
            }
    }

    private fun parsePrimer(
        raw: String,
        fallback: String,
    ): PrimerParsed {
        return runCatching {
            val element = Json.parseToJsonElement(raw)
            val obj = element as? JsonObject ?: return PrimerParsed(fallback, emptyList(), null)
            val answer = obj["intermediate_answer"]?.jsonPrimitive?.content ?: fallback
            val followUps =
                (obj["follow_up_queries"] as? JsonArray)
                    ?.mapNotNull { it.jsonPrimitive.contentOrNull }
                    .orEmpty()
            val score = obj["score"]?.jsonPrimitive?.doubleOrNull
            PrimerParsed(answer, followUps, score)
        }.getOrElse { PrimerParsed(fallback, emptyList(), null) }
    }

    private data class PrimerParsed(
        val answer: String,
        val followUps: List<String>,
        val score: Double?,
    )

    private data class DriftAction(
        val followUpQuery: String,
        var result: QueryResult? = null,
    )

    private data class DriftQueryState(
        val pending: MutableList<DriftAction> = mutableListOf(),
        val completedActions: MutableList<DriftAction> = mutableListOf(),
        val seenQueries: MutableSet<String> = mutableSetOf(),
    ) {
        fun addActions(followUps: List<String>) {
            followUps
                .filter { it.isNotBlank() }
                .filter { seenQueries.add(it) }
                .forEach { pending.add(DriftAction(it)) }
        }

        fun hasPendingActions(): Boolean = pending.any { it.result == null }

        fun nextPendingAction(): DriftAction? =
            pending
                .filter { it.result == null }
                .maxByOrNull { it.result?.score ?: 0.0 } // prioritize higher scored if available
                ?: pending.firstOrNull { it.result == null }
    }

    private fun buildPrimerFallback(question: String): QueryResult {
        val global = runBlocking { globalSearchEngine?.search(question) }
        if (global != null) {
            return QueryResult(
                answer = global.answer,
                context = emptyList(),
                contextRecords = global.contextRecords,
                contextText = global.reduceContextText,
                llmCalls = global.llmCalls,
                promptTokens = global.promptTokens,
                outputTokens = global.outputTokens,
                llmCallsCategories = global.llmCallsCategories,
                promptTokensCategories = global.promptTokensCategories,
                outputTokensCategories = global.outputTokensCategories,
            )
        }
        return QueryResult("No primer available.", emptyList())
    }

    companion object {
        private val DRIFT_PRIMER_PROMPT =
            """
            You are a helpful agent designed to reason over a knowledge graph in response to a user query.
            This is a unique knowledge graph where edges are freeform text rather than verb operators. You will begin your reasoning looking at a summary of the content of the most relevant communites and will provide:

            1. score: How well the intermediate answer addresses the query. A score of 0 indicates a poor, unfocused answer, while a score of 100 indicates a highly focused, relevant answer that addresses the query in its entirety.

            2. intermediate_answer: This answer should match the level of detail and length found in the community summaries. The intermediate answer should be exactly 2000 characters long. This must be formatted in markdown and must begin with a header that explains how the following text is related to the query.

            3. follow_up_queries: A list of follow-up queries that could be asked to further explore the topic. These should be formatted as a list of strings. Generate at least five good follow-up queries.

            Use this information to help you decide whether or not you need more information about the entities mentioned in the report. You may also use your general knowledge to think of entities which may help enrich your answer.

            You will also provide a full answer from the content you have available. Use the data provided to generate follow-up queries to help refine your search. Do not ask compound questions, for example: "What is the market cap of Apple and Microsoft?". Use your knowledge of the entity distribution to focus on entity types that will be useful for searching a broad area of the knowledge graph.

            For the query:

            {query}

            The top-ranked community summaries:

            {community_reports}

            Provide the intermediate answer, and all scores in JSON format following:

            {{'intermediate_answer': str,
            'score': int,
            'follow_up_queries': List[str]}}

            Begin:
            """.trimIndent()
        private val DRIFT_REDUCE_PROMPT =
            """
            ---Role---

            You are a helpful assistant responding to questions about data in the reports provided.

            ---Goal---

            Generate a response of the target length and format that responds to the user's question, summarizing all information in the input reports appropriate for the response length and format, and incorporating any relevant general knowledge while being as specific, accurate and concise as possible.

            If you don't know the answer, just say so. Do not make anything up.

            Points supported by data should list their data references as follows:

            "This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

            Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

            For example:

            "Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Sources (1, 5, 15)]."

            Do not include information where the supporting evidence for it is not provided.

            If you decide to use general knowledge, you should add a delimiter stating that the information is not supported by the data tables. For example:

            "Person X is the owner of Company Y and subject to many allegations of wrongdoing. [Data: General Knowledge (href)]"

            ---Data Reports---

            {context_data}

            ---Target response length and format---

            {response_type}


            ---Goal---

            Generate a response of the target length and format that responds to the user's question, summarizing all information in the input reports appropriate for the response length and format, and incorporating any relevant general knowledge while being as specific, accurate and concise as possible.

            If you don't know the answer, just say so. Do not make anything up.

            Points supported by data should list their data references as follows:

            "This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

            Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

            For example:

            "Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Sources (1, 5, 15)]."

            Do not include information where the supporting evidence for it is not provided.

            If you decide to use general knowledge, you should add a delimiter stating that the information is not supported by the data tables. For example:

            "Person X is the owner of Company Y and subject to many allegations of wrongdoing. [Data: General Knowledge (href)]".

            Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown. Now answer the following query using the data above:
            """.trimIndent()
    }
}
