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
    val contextRecords: Map<String, List<Map<String, String>>> = emptyMap(),
    val contextText: String = "",
)

private data class DriftNode(
    val id: Int,
    val query: String,
    val parentId: Int?,
    val type: NodeType,
    var result: QueryResult? = null,
) {
    enum class NodeType {
        PRIMER,
        FOLLOW_UP,
        PROVIDED,
    }
}

private class DriftSearchState {
    private val nodes = mutableListOf<DriftNode>()
    private val pending = ArrayDeque<Int>()
    private val seenQueries = mutableSetOf<String>()
    private var nextId = 0

    fun addPrimer(
        query: String,
        primerResult: QueryResult,
        followUps: List<String>,
    ) {
        val node = DriftNode(nextId++, query, null, DriftNode.NodeType.PRIMER, primerResult)
        nodes += node
        seenQueries += query.lowercase()
        enqueueFollowUps(node.id, followUps.ifEmpty { primerResult.followUpQueries.ifEmpty { listOf(query) } })
    }

    fun enqueueFollowUps(
        parentId: Int?,
        followUps: List<String>,
    ) {
        followUps
            .map { it.trim() }
            .filter { it.isNotBlank() }
            .forEach { followUp ->
                val key = followUp.lowercase()
                if (seenQueries.add(key)) {
                    val node = DriftNode(nextId++, followUp, parentId, DriftNode.NodeType.FOLLOW_UP)
                    nodes += node
                    pending.add(node.id)
                }
            }
    }

    fun hasPending(): Boolean = pending.any { id -> findNode(id)?.result == null }

    fun nextActions(limit: Int): List<DriftNode> {
        val actions = mutableListOf<DriftNode>()
        while (pending.isNotEmpty() && actions.size < limit) {
            val nextId = pending.removeFirst()
            val node = findNode(nextId) ?: continue
            if (node.result == null) actions += node
        }
        return actions
    }

    fun markResult(
        node: DriftNode,
        result: QueryResult,
    ) {
        node.result = result
        enqueueFollowUps(node.id, result.followUpQueries)
    }

    fun completedResults(): List<QueryResult> = nodes.mapNotNull { it.result }

    fun contextText(): String {
        val ordered = nodes.filter { it.result != null }.sortedBy { it.id }
        return ordered.joinToString("\n\n") { node ->
            buildString {
                val label =
                    when (node.type) {
                        DriftNode.NodeType.PRIMER -> "Primer"
                        DriftNode.NodeType.PROVIDED -> "Provided Follow-up"
                        DriftNode.NodeType.FOLLOW_UP -> "Follow-up"
                    }
                append("----$label (id=${node.id})----\n")
                append("Query: ${node.query}\n")
                node.result?.score?.let { append("Score: $it\n") }
                append(node.result?.answer ?: "")
            }
        }
    }

    fun contextRecords(): Map<String, List<MutableMap<String, String>>> {
        val actionRecords =
            nodes.map { node ->
                mutableMapOf(
                    "id" to node.id.toString(),
                    "parent_id" to (node.parentId?.toString() ?: ""),
                    "query" to node.query,
                    "type" to node.type.name.lowercase(),
                    "score" to (node.result?.score?.toString() ?: ""),
                    "in_context" to if (node.result != null) "true" else "false",
                ).apply {
                    node.result?.answer?.let { this["answer"] = it }
                }
            }
        return mapOf("actions" to actionRecords.toMutableList())
    }

    private fun findNode(id: Int): DriftNode? = nodes.firstOrNull { it.id == id }
}

private data class PlannerResult(
    val state: DriftSearchState,
    val primerResult: QueryResult,
    val localResults: List<QueryResult>,
    val contextText: String,
    val contextRecords: Map<String, List<MutableMap<String, String>>>,
    val llmCalls: Int,
    val promptTokens: Int,
    val outputTokens: Int,
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
    private val primerSystemPrompt: String = DEFAULT_DRIFT_PRIMER_PROMPT,
    private val reduceSystemPrompt: String = DEFAULT_DRIFT_REDUCE_PROMPT,
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
        val planner = runPlanner(question, followUpQueries)
        callbacks.forEach { it.onContext(planner.contextRecords) }

        val reduceResult = reduce(question, planner.contextText)

        val primerCalls = planner.primerResult.llmCalls
        val primerPromptTokens = planner.primerResult.promptTokens
        val primerOutputTokens = planner.primerResult.outputTokens

        val localCalls = planner.localResults.sumOf { it.llmCalls }
        val localPromptTokens = planner.localResults.sumOf { it.promptTokens }
        val localOutputTokens = planner.localResults.sumOf { it.outputTokens }

        val llmCallsCategories =
            mapOf(
                "primer" to primerCalls,
                "local" to localCalls,
                "reduce" to reduceResult.llmCalls,
            )
        val promptTokensCategories =
            mapOf(
                "primer" to primerPromptTokens,
                "local" to localPromptTokens,
                "reduce" to reduceResult.promptTokens,
            )
        val outputTokensCategories =
            mapOf(
                "primer" to primerOutputTokens,
                "local" to localOutputTokens,
                "reduce" to reduceResult.outputTokens,
            )

        val totalLlmCalls = llmCallsCategories.values.sum()
        val totalPromptTokens = promptTokensCategories.values.sum()
        val totalOutputTokens = outputTokensCategories.values.sum()

        val allActions = planner.state.completedResults()
        val finalAnswer = reduceResult.answer.ifBlank { allActions.lastOrNull()?.answer ?: "" }

        return DriftSearchResult(
            answer = finalAnswer,
            actions = allActions,
            llmCalls = totalLlmCalls,
            promptTokens = totalPromptTokens,
            outputTokens = totalOutputTokens,
            llmCallsCategories = llmCallsCategories,
            promptTokensCategories = promptTokensCategories,
            outputTokensCategories = outputTokensCategories,
            contextRecords = planner.contextRecords.toImmutableContextRecords(),
            contextText = planner.contextText,
        )
    }

    private suspend fun runPlanner(
        question: String,
        followUpQueries: List<String>,
    ): PlannerResult {
        var totalLlmCalls = 0
        var totalPromptTokens = 0
        var totalOutputTokens = 0

        val primer =
            try {
                primerSearch(question)
            } catch (error: Exception) {
                buildPrimerFallback(question)
            }
        totalLlmCalls += primer.llmCalls
        totalPromptTokens += primer.promptTokens
        totalOutputTokens += primer.outputTokens

        val state = DriftSearchState()
        state.addPrimer(question, primer, followUpQueries)

        val localResults = mutableListOf<QueryResult>()
        var iteration = 0
        while (state.hasPending() && iteration < maxIterations) {
            val batch = state.nextActions(maxIterations - iteration)
            if (batch.isEmpty()) break
            for (node in batch) {
                val local =
                    localQueryEngine.answer(
                        question = node.query,
                        responseType = responseType,
                        driftQuery = question,
                    )
                state.markResult(node, local)
                localResults += local
                totalLlmCalls += local.llmCalls
                totalPromptTokens += local.promptTokens
                totalOutputTokens += local.outputTokens
                iteration++
                if (iteration >= maxIterations) break
            }
        }

        val contextText = state.contextText()
        val mergedRecords = mutableMapOf<String, MutableList<MutableMap<String, String>>>()

        fun mergeRecords(source: Map<String, List<Map<String, String>>>) {
            source.forEach { (key, records) ->
                val dest = mergedRecords.getOrPut(key) { mutableListOf() }
                dest += records.map { it.toMutableMap() }
            }
        }
        mergeRecords(state.contextRecords())
        mergeRecords(primer.contextRecords)
        localResults.forEach { mergeRecords(it.contextRecords) }
        val contextRecords = mergedRecords
        return PlannerResult(
            state = state,
            primerResult = primer,
            localResults = localResults,
            contextText = contextText,
            contextRecords = contextRecords,
            llmCalls = totalLlmCalls,
            promptTokens = totalPromptTokens,
            outputTokens = totalOutputTokens,
        )
    }

    fun streamSearch(
        question: String,
        followUpQueries: List<String> = emptyList(),
    ): Flow<String> =
        callbackFlow {
            val planner = runBlocking { runPlanner(question, followUpQueries) }
            callbacks.forEach { it.onContext(planner.contextRecords) }

            planner.state.completedResults().forEach { res ->
                if (res.answer.isNotBlank()) trySend(res.answer)
            }

            val contextText = planner.contextText
            val prompt =
                reduceSystemPrompt
                    .replace("{context_data}", contextText)
                    .replace("{response_type}", responseType)
                    .let { base ->
                        if (reduceParams.jsonResponse) "$base\nReturn ONLY valid JSON per the schema above." else base
                    }
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

    private fun reduce(
        question: String,
        contextText: String,
    ): QueryResult {
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
                .replace("{community_reports}", context.text)
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
            contextRecords = mapOf("reports" to context.records.map { it.toMap() }),
            contextText = context.text,
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

    private fun buildPrimerContext(topK: Int = 5): PrimerContext {
        if (communityReports.isEmpty()) return PrimerContext("", emptyList())
        val sorted = communityReports.sortedByDescending { it.rank ?: 0.0 }
        val selected = sorted.take(topK)
        val text =
            selected.joinToString("\n\n") { report ->
                "Community ${report.communityId} (rank=${report.rank ?: 0.0}): ${report.summary}"
            }
        val records =
            sorted.map { report ->
                val id = report.id ?: report.shortId ?: report.communityId.toString()
                mutableMapOf(
                    "id" to id,
                    "community_id" to report.communityId.toString(),
                    "title" to (report.title ?: report.communityId.toString()),
                    "rank" to (report.rank?.toString() ?: ""),
                    "in_context" to if (report in selected) "true" else "false",
                )
            }
        return PrimerContext(text, records)
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

    private data class PrimerContext(
        val text: String,
        val records: List<MutableMap<String, String>>,
    )

    private suspend fun buildPrimerFallback(question: String): QueryResult {
        val global = globalSearchEngine?.search(question)
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
        internal val DEFAULT_DRIFT_PRIMER_PROMPT =
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
        internal val DEFAULT_DRIFT_REDUCE_PROMPT =
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

            Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown. Now answer the following query using the data above:
            """.trimIndent()
    }
}
