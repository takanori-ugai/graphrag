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

    /**
     * Adds a primer node to the search state and enqueues its follow-up queries.
     *
     * The follow-up queries enqueued are chosen in this order of precedence: the supplied `followUps` list if non-empty, otherwise `primerResult.followUpQueries` if non-empty, otherwise a single follow-up equal to `query`.
     *
     * @param query The primer query text.
     * @param primerResult The result produced for the primer node.
     * @param followUps Optional explicit follow-up queries to enqueue for this primer.
     */
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

    /**
     * Adds cleaned, deduplicated follow-up queries as FOLLOW_UP nodes and enqueues them for processing.
     *
     * Trims each follow-up, ignores blank entries, and deduplicates case-insensitively against previously seen queries.
     * For each new follow-up a DriftNode of type FOLLOW_UP is created with the given parentId, appended to the internal
     * node list, and its id is added to the pending queue.
     *
     * @param parentId The id of the parent node for these follow-ups, or `null` if there is no parent.
     * @param followUps A list of follow-up query strings to clean, deduplicate, and enqueue.
     */
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

    /**
 * Indicates whether there are any pending nodes that have not yet produced a result.
 *
 * @return `true` if at least one pending node's result is null, `false` otherwise.
 */
fun hasPending(): Boolean = pending.any { id -> findNode(id)?.result == null }

    /**
     * Dequeues up to `limit` pending action nodes and returns those that still lack results.
     *
     * This removes inspected node ids from the internal pending queue; nodes whose `result`
     * is non-null are skipped and not included in the returned list.
     *
     * @param limit Maximum number of pending nodes with null results to return.
     * @return A list of pending `DriftNode` objects (in queue order) that have no result, up to `limit`.
     */
    fun nextActions(limit: Int): List<DriftNode> {
        val actions = mutableListOf<DriftNode>()
        while (pending.isNotEmpty() && actions.size < limit) {
            val nextId = pending.removeFirst()
            val node = findNode(nextId) ?: continue
            if (node.result == null) actions += node
        }
        return actions
    }

    /**
     * Assigns a query result to the given node and enqueues any follow-up queries from that result.
     *
     * @param node The node to attach the result to.
     * @param result The query result whose follow-up queries will be enqueued.
     */
    fun markResult(
        node: DriftNode,
        result: QueryResult,
    ) {
        node.result = result
        enqueueFollowUps(node.id, result.followUpQueries)
    }

    /**
 * Collects completed query results from all nodes.
 *
 * @return A list of `QueryResult` objects for nodes that have a non-null result, in the order the nodes were added.
 */
fun completedResults(): List<QueryResult> = nodes.mapNotNull { it.result }

    /**
     * Builds a textual context of all completed nodes in ascending id order.
     *
     * Each completed node is rendered as a block with a header `----<Label> (id=<id>)----`,
     * followed by `Query: <query>`, an optional `Score: <score>` line if present, and the node answer.
     *
     * @return A single string containing all node blocks separated by blank lines.
     */
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

    /**
     * Builds structured records for all nodes, suitable for downstream consumption or telemetry.
     *
     * Each record describes a node's id, parent, query text, node type, score, and whether it contributed to the context; includes the node's answer when present.
     *
     * @return A map with the key `"actions"` mapping to a list of mutable maps. Each map contains:
     *  - `"id"`: node id as a string
     *  - `"parent_id"`: parent id as a string or empty string if none
     *  - `"query"`: the node's query text
     *  - `"type"`: node type name in lowercase (`"primer"`, `"follow_up"`, or `"provided"`)
     *  - `"score"`: node result score as a string or empty string if unavailable
     *  - `"in_context"`: `"true"` if the node has a result, `"false"` otherwise
     *  - optional `"answer"`: the node's answer text when a result exists
     */
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

    /**
 * Locate a node by its identifier.
 *
 * @param id The node's numeric identifier.
 * @return The matching [DriftNode], or `null` if no node with the given id exists.
 */
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
    /**
     * Performs a DRIFT-style search for the given question, producing a final answer, collected action results, and usage accounting.
     *
     * Notifies configured callbacks with planner context records, runs a reduction step over accumulated context, and aggregates LLM call and token usage by stage.
     *
     * @param question The user question to answer.
     * @param followUpQueries Optional initial follow-up queries to seed the planner.
     * @return A DriftSearchResult containing the final answer (or the last action's answer if the reducer is blank), the list of action results, context text and records, and aggregated LLM call and token metrics broken down by stage.
     */
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

    /**
     * Executes the planning phase: runs the primer (with fallback), expands and executes local follow-up queries up to the iteration limit, and assembles planner state and usage totals.
     *
     * @param question The main user question driving the primer and local queries.
     * @param followUpQueries Optional follow-up queries to seed the planner in addition to those produced by the primer.
     * @return A PlannerResult containing the constructed DriftSearchState, the primer result, collected local results, merged context text and records, and aggregated LLM usage counters (llmCalls, promptTokens, outputTokens).
     */
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

    /**
         * Streams a DRIFT-style search that first emits completed local action answers and then streams the reduce-phase response.
         *
         * Runs planning synchronously, emits each non-blank completed action answer, then starts a streaming reduce prompt and emits partial tokens as they arrive. The flow closes when the reduce response completes and will close with an error if the streaming model reports an error.
         *
         * @param question The user's question to answer.
         * @param followUpQueries Optional initial follow-up queries to seed the planner.
         * @return A Flow of strings containing emitted completed action answers (if any) followed by partial reduce-phase tokens; the flow completes when the reduce step finishes.
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

    /**
     * Sends a reduce prompt (including provided context and the user question) to the streaming model and returns a QueryResult representing the reduce-phase response.
     *
     * @param question The user's question to include in the reduce prompt.
     * @param contextText The textual context assembled from prior planner/local results to include in the prompt.
     * @return A QueryResult containing the reduce-phase answer, empty context lists/records, the supplied contextText, and usage counts (llmCalls, promptTokens, outputTokens) categorized under "reduce". The `answer` will be an empty string if the streaming model fails or an error occurs while collecting the response.
     */
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

    /**
     * Produces an initial "primer" result for the given question by consulting community reports and generating
     * an initial answer, follow-up queries, and usage metrics for the primer stage.
     *
     * @param question The user's question to prime the search.
     * @return A QueryResult containing the primer answer, extracted follow-up queries, optional score,
     *         text and record-based context derived from community reports, and token/LLM call accounting
     *         categorized under the "primer" stage.
     */
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

    /**
     * Builds a compact primer context from the top community reports.
     *
     * Produces a text blob concatenating up to `topK` reports (sorted by descending rank) and a list of record maps for all reports indicating whether each was included in the text.
     *
     * @param topK Maximum number of highest-ranked reports to include in the returned text; defaults to 5.
     * @return A [PrimerContext] containing `text` with selected report summaries and `records` where each map contains keys: `"id"`, `"community_id"`, `"title"`, `"rank"`, and `"in_context"` (`"true"` if the report appears in the `text`, otherwise `"false"`).
     */
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

    /**
     * Parses a primer LLM output into its intermediate answer, follow-up queries, and score.
     *
     * @param raw The raw primer output (typically JSON).
     * @param fallback The fallback answer to use when parsing fails or the answer field is missing.
     * @return A [PrimerParsed] containing the parsed `answer`, `followUps`, and `score`. If parsing fails, returns a `PrimerParsed` with `answer` set to `fallback`, an empty `followUps` list, and `score` set to `null`.
     */
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

    /**
     * Obtain a primer QueryResult by delegating to the configured global search engine or returning a default fallback.
     *
     * @param question The question used to perform the global search if a global search engine is configured.
     * @return A QueryResult created from the global search result preserving answer, context records, context text, and usage counters when available; otherwise a QueryResult with the answer "No primer available." and empty context.
     */
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