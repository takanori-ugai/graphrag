package com.microsoft.graphrag.query

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingType
import com.microsoft.graphrag.index.CommunityReport
import dev.langchain4j.model.chat.response.ChatResponse
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler
import dev.langchain4j.model.openai.OpenAiStreamingChatModel
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit
import kotlin.math.min

data class GlobalSearchResult(
    val answer: String,
    val mapResponses: List<QueryResult>,
    val reduceContextText: String,
    val contextRecords: Map<String, List<Map<String, String>>>,
    val llmCalls: Int,
    val promptTokens: Int,
    val outputTokens: Int,
    val llmCallsCategories: Map<String, Int>,
    val promptTokensCategories: Map<String, Int>,
    val outputTokensCategories: Map<String, Int>,
)

private data class ContextBuildResult(
    val chunks: List<String>,
    val contextRecords: Map<String, List<MutableMap<String, String>>>,
    val llmCalls: Int,
    val promptTokens: Int,
    val outputTokens: Int,
)

private data class SelectionResult(
    val reports: List<CommunityReport>,
    val contextRecords: Map<String, List<MutableMap<String, String>>>,
    val llmCalls: Int,
    val promptTokens: Int,
    val outputTokens: Int,
)

private data class RatingResult(
    val communityId: Int,
    val rating: Int,
    val promptTokens: Int,
    val outputTokens: Int,
)

/**
 * Global search engine that mirrors Python's map-reduce flow over community reports.
 */
class GlobalSearchEngine(
    private val streamingModel: OpenAiStreamingChatModel,
    private val communityReports: List<CommunityReport>,
    private val communityHierarchy: Map<Int, Int> = emptyMap(),
    private val communityLevel: Int? = null,
    private val dynamicCommunitySelection: Boolean = false,
    private val dynamicThreshold: Int = 1,
    private val dynamicKeepParent: Boolean = false,
    private val dynamicNumRepeats: Int = 1,
    private val dynamicUseSummary: Boolean = false,
    private val dynamicMaxLevel: Int = 2,
    private val callbacks: List<QueryCallbacks> = emptyList(),
    private val mapSystemPrompt: String = DEFAULT_MAP_SYSTEM_PROMPT,
    private val reduceSystemPrompt: String = DEFAULT_REDUCE_SYSTEM_PROMPT,
    private val responseType: String = "multiple paragraphs",
    private val allowGeneralKnowledge: Boolean = false,
    private val generalKnowledgeInstruction: String = DEFAULT_GENERAL_KNOWLEDGE_INSTRUCTION,
    private val ratingPrompt: String = DEFAULT_COMMUNITY_RATING_PROMPT,
    private val mapMaxLength: Int = 1_000,
    private val reduceMaxLength: Int = 2_000,
    private val maxContextTokens: Int = 8_000,
    private val maxDataTokens: Int = maxContextTokens,
    private val mapParams: ModelParams = ModelParams(jsonResponse = true),
    private val reduceParams: ModelParams = ModelParams(jsonResponse = false),
    private val encoding: Encoding = Encodings.newLazyEncodingRegistry().getEncoding(EncodingType.CL100K_BASE),
) {
    /**
     * Performs a map-reduce search over the configured community reports and returns the aggregated result.
     *
     * @param question The user query to search for.
     * @param conversationHistory Ordered list of previous conversation turns, earliest first, to include as context.
     * @return A [GlobalSearchResult] containing the final answer, the list of per-chunk map responses, the reduced context text, context records, total LLM call and token counts, and per-phase category breakdowns for build, map, and reduce.
     */
    suspend fun search(
        question: String,
        conversationHistory: List<String> = emptyList(),
    ): GlobalSearchResult =
        coroutineScope {
            val contextResult = buildContextChunks(question, conversationHistory)
            callbacks.forEach { it.onContext(contextResult.contextRecords) }
            callbacks.forEach { it.onMapResponseStart(contextResult.chunks) }

            val mapResponses =
                contextResult.chunks
                    .map { chunk -> async { mapStep(question, chunk) } }
                    .awaitAll()
            callbacks.forEach { it.onMapResponseEnd(mapResponses) }

            val reduceResult = reduceStep(question, mapResponses, conversationHistory)

            val llmCallsCategories =
                mapOf(
                    "build_context" to contextResult.llmCalls,
                    "map" to mapResponses.sumOf { it.llmCalls },
                    "reduce" to reduceResult.llmCalls,
                )
            val promptTokensCategories =
                mapOf(
                    "build_context" to contextResult.promptTokens,
                    "map" to mapResponses.sumOf { it.promptTokens },
                    "reduce" to reduceResult.promptTokens,
                )
            val outputTokensCategories =
                mapOf(
                    "build_context" to contextResult.outputTokens,
                    "map" to mapResponses.sumOf { it.outputTokens },
                    "reduce" to reduceResult.outputTokens,
                )

            GlobalSearchResult(
                answer = reduceResult.answer,
                mapResponses = mapResponses,
                reduceContextText = reduceResult.contextText,
                contextRecords = contextResult.contextRecords.toImmutableContextRecords(),
                llmCalls = llmCallsCategories.values.sum(),
                promptTokens = promptTokensCategories.values.sum(),
                outputTokens = outputTokensCategories.values.sum(),
                llmCallsCategories = llmCallsCategories,
                promptTokensCategories = promptTokensCategories,
                outputTokensCategories = outputTokensCategories,
            )
        }

    /**
     * Builds textual context chunks and associated metadata for the given question and conversation history.
     *
     * Uses report selection to produce one or more context text chunks (conversation section followed by a table of report rows)
     * that each respect the configured maximum context token budget, and assembles context records and LLM token/usage totals
     * for the context-building phase.
     *
     * @param question The user's query driving report selection and context construction.
     * @param conversationHistory Ordered list of prior conversation turns to include in the context.
     * @return A ContextBuildResult containing the list of context text chunks, a map of context records (reports and conversation),
     *         and aggregated LLM usage counts (llmCalls, promptTokens, outputTokens) attributable to the context-building step.
     */
    private suspend fun buildContextChunks(
        question: String,
        conversationHistory: List<String>,
    ): ContextBuildResult {
        if (communityReports.isEmpty()) return ContextBuildResult(emptyList(), emptyMap(), 0, 0, 0)
        val selection = selectReports(question, conversationHistory)
        val (conversationSection, conversationRecords) = buildConversationSection(conversationHistory)
        val header = listOf("id", "title", "summary", "rank").joinToString("|")
        val ratingLookup = selection.contextRecords["reports"]?.associateBy { it["id"] ?: "" }.orEmpty()

        val chunks = mutableListOf<String>()
        var current =
            StringBuilder().apply {
                if (conversationSection.isNotBlank()) {
                    append(conversationSection).append("\n\n")
                }
                append("-----Reports-----\n$header\n")
            }
        var tokens = tokenCount(current.toString())
        val records = mutableListOf<MutableMap<String, String>>()
        for (report in selection.reports) {
            val rowParts =
                listOf(
                    report.id ?: report.communityId.toString(),
                    report.title ?: report.communityId.toString(),
                    report.summary,
                    (report.rank ?: 0.0).toString(),
                )
            val row = rowParts.joinToString("|") + "\n"
            val rowTokens = tokenCount(row)
            if (tokens + rowTokens > maxContextTokens && current.isNotEmpty()) {
                chunks += current.toString().trimEnd()
                current = StringBuilder("-----Reports-----\n$header\n")
                tokens = tokenCount(current.toString())
            }
            current.append(row)
            tokens += rowTokens
            records +=
                header
                    .split("|")
                    .zip(rowParts)
                    .toMap()
                    .toMutableMap()
                    .apply {
                        this["in_context"] = "true"
                        ratingLookup[report.communityId.toString()]?.get("rating")?.let { this["rating"] = it }
                    }
        }
        if (current.isNotEmpty()) chunks += current.toString().trimEnd()
        val contextRecords =
            selection.contextRecords.toMutableMap().apply {
                if (records.isNotEmpty()) {
                    this["reports"] = records
                }
                if (conversationRecords.isNotEmpty()) {
                    this["conversation"] = conversationRecords.toMutableList()
                }
            }
        val chunkTokens = chunks.sumOf { tokenCount(it) }
        return ContextBuildResult(
            chunks = chunks,
            contextRecords = contextRecords,
            llmCalls = selection.llmCalls,
            promptTokens = selection.promptTokens + chunkTokens,
            outputTokens = selection.outputTokens,
        )
    }

    /**
     * Selects which community reports should be included in the search context.
     *
     * When dynamicCommunitySelection is disabled, returns the filtered reports sorted by rank.
     * When enabled, iteratively rates communities starting from the top level up to dynamicMaxLevel,
     * includes communities whose rating meets or exceeds dynamicThreshold, and optionally promotes
     * or removes parents according to dynamicKeepParent. Accumulates per-report rating metadata and
     * LLM usage metrics incurred during the selection process.
     *
     * @param question The user question used to evaluate report relevance; conversationHistory is appended to this.
     * @param conversationHistory Past conversation turns that are appended to the question for relevance evaluation.
     * @return A SelectionResult containing the selected reports (sorted by rank), a map of rating context records
     *         (under the "reports" key when present), and aggregated LLM usage metrics: llmCalls, promptTokens,
     *         and outputTokens.
     */
    private suspend fun selectReports(
        question: String,
        conversationHistory: List<String>,
    ): SelectionResult {
        val enrichedQuestion = appendConversationHistory(question, conversationHistory)
        val filtered =
            if (communityLevel != null) {
                communityReports.filter { levelOf(it.communityId) == communityLevel }
            } else {
                communityReports
            }
        if (!dynamicCommunitySelection) {
            val sorted = filtered.sortedByDescending { it.rank ?: 0.0 }
            return SelectionResult(sorted, emptyMap(), 0, 0, 0)
        }

        val reportsById = filtered.associateBy { it.communityId }
        if (reportsById.isEmpty()) return SelectionResult(emptyList(), emptyMap(), 0, 0, 0)
        val maxLevel = communityLevel?.let { min(it, dynamicMaxLevel) } ?: dynamicMaxLevel
        val levels = reportsById.values.groupBy { levelOf(it.communityId) }
        var queue = levels[0]?.map { it.communityId } ?: emptyList()
        var level = 0
        val ratings = mutableMapOf<Int, Int>()
        var llmCalls = 0
        var promptTokens = 0
        var outputTokens = 0
        val selected = mutableSetOf<Int>()
        val children = mutableMapOf<Int, MutableList<Int>>()
        communityHierarchy.forEach { (child, parent) ->
            children.getOrPut(parent) { mutableListOf() }.add(child)
        }

        while (queue.isNotEmpty() && level <= maxLevel) {
            val results =
                coroutineScope {
                    queue
                        .map { id ->
                            val report = reportsById[id]
                            async {
                                report?.let { rateCommunity(enrichedQuestion, it) }
                            }
                        }.awaitAll()
                }.filterNotNull()

            val next = mutableListOf<Int>()
            for (result in results) {
                ratings[result.communityId] = result.rating
                llmCalls += 1
                promptTokens += result.promptTokens
                outputTokens += result.outputTokens
                if (result.rating >= dynamicThreshold) {
                    selected += result.communityId
                    children[result.communityId]?.filter { reportsById.containsKey(it) }?.let { next.addAll(it) }
                    if (!dynamicKeepParent) {
                        communityHierarchy[result.communityId]?.let { parent -> selected.remove(parent) }
                    }
                }
            }
            queue = next
            level += 1
            if (queue.isEmpty() && selected.isEmpty() && level <= maxLevel) {
                queue = levels[level]?.map { it.communityId } ?: emptyList()
            }
        }

        val selectedReports =
            if (selected.isNotEmpty()) {
                selected.mapNotNull { reportsById[it] }
            } else {
                filtered.sortedByDescending { it.rank ?: 0.0 }
            }
        val ratingRecords =
            if (ratings.isNotEmpty()) {
                mapOf(
                    "reports" to
                        ratings.map { (id, rating) ->
                            mutableMapOf(
                                "id" to id.toString(),
                                "rating" to rating.toString(),
                                "in_context" to if (id in selected) "true" else "false",
                            )
                        },
                )
            } else {
                emptyMap()
            }

        return SelectionResult(
            reports = selectedReports.sortedByDescending { it.rank ?: 0.0 },
            contextRecords = ratingRecords,
            llmCalls = llmCalls,
            promptTokens = promptTokens,
            outputTokens = outputTokens,
        )
    }

    /**
     * Computes the hierarchical depth of a community within the configured communityHierarchy.
     *
     * Traverses parent links from the given communityId up to the root (a missing or negative parent)
     * and returns the number of steps from the community to that root.
     *
     * @param communityId The identifier of the community whose level to compute.
     * @return The number of ancestor links between the community and the root (root communities return 0).
     */
    private fun levelOf(communityId: Int): Int {
        var current: Int? = communityId
        var level = 0
        while (current != null) {
            val parent = communityHierarchy[current]
            if (parent == null || parent < 0) break
            level += 1
            current = parent
        }
        return level
    }

    /**
     * Produces a numeric rating for a community report relative to the provided question and returns that rating with token usage metrics.
     *
     * Sends the configured rating prompt (possibly repeated) for the report and aggregates the most frequent numeric rating across repeats.
     *
     * @param question The user question used to contextualize the rating.
     * @param report The CommunityReport being rated.
     * @return A RatingResult containing the report's communityId, the chosen rating (mode of collected ratings, defaulting to 1 if unavailable), and the accumulated prompt and output token counts.
     */
    private suspend fun rateCommunity(
        question: String,
        report: CommunityReport,
    ): RatingResult {
        val description =
            if (dynamicUseSummary || report.fullContent.isNullOrBlank()) {
                report.summary
            } else {
                report.fullContent ?: report.summary
            }
        val prompt =
            ratingPrompt
                .replace("{description}", description)
                .replace("{question}", question)
        var promptTokens = 0
        var outputTokens = 0
        val ratings = mutableListOf<Int>()
        repeat(dynamicNumRepeats.coerceAtLeast(1)) {
            promptTokens += tokenCount(prompt)
            val answer = streamAnswer(prompt)
            outputTokens += tokenCount(answer)
            val rating =
                runCatching {
                    val element = Json.parseToJsonElement(answer)
                    (element as? JsonObject)
                        ?.get("rating")
                        ?.jsonPrimitive
                        ?.content
                        ?.toIntOrNull()
                }.getOrNull() ?: 1
            ratings += rating
        }
        val rating =
            ratings
                .groupingBy { it }
                .eachCount()
                .maxByOrNull { it.value }
                ?.key ?: 1
        return RatingResult(
            communityId = report.communityId,
            rating = rating,
            promptTokens = promptTokens,
            outputTokens = outputTokens,
        )
    }

    /**
     * Generates a per-chunk "map" response for the given question using the provided context.
     *
     * @return `QueryResult` containing the map-phase answer, minimal context records, and LLM/token usage tallied under the "map" category.
     */
    private suspend fun mapStep(
        question: String,
        context: String,
    ): QueryResult {
        val prompt =
            mapSystemPrompt
                .replace("{context_data}", context)
                .replace("{max_length}", mapMaxLength.toString())
                .let { base ->
                    if (mapParams.jsonResponse) "$base\nReturn ONLY valid JSON per the schema above." else base
                }
        val promptTokens = tokenCount(prompt)
        val fullPrompt = "$prompt\n\nUser question: $question"
        val answerText = streamAnswer(fullPrompt)
        val outputTokens = tokenCount(answerText)
        return QueryResult(
            answer = answerText,
            context = emptyList(),
            contextRecords = mapOf("reports" to listOf(mapOf("in_context" to "true"))),
            llmCalls = 1,
            promptTokens = promptTokens,
            outputTokens = outputTokens,
            llmCallsCategories = mapOf("map" to 1),
            promptTokensCategories = mapOf("map" to promptTokens),
            outputTokensCategories = mapOf("map" to outputTokens),
        )
    }

    /**
     * Synthesizes a final answer from per-chunk map responses and optional conversation history.
     *
     * Builds a reduce prompt from key points extracted from `mapResponses` (keeps points with score > 0 unless
     * general knowledge is allowed), includes the conversation history, and streams a final reduce answer
     * while accounting tokens and LLM call counts. If no usable points remain and general knowledge is
     * disallowed, returns a sentinel `NO_DATA_ANSWER` with zero usage metrics.
     *
     * @param question The user question to answer.
     * @param mapResponses The list of map-phase `QueryResult`s whose extracted points will be considered for reduction.
     * @param conversationHistory Historical turns to include in the reduce context.
     * @return A `QueryResult` containing the synthesized answer, the composed reduce context text, and token/LLM usage categorized for the reduce phase (or a `NO_DATA_ANSWER` result with zero usage when no data is usable).
     */
    private suspend fun reduceStep(
        question: String,
        mapResponses: List<QueryResult>,
        conversationHistory: List<String>,
    ): QueryResult {
        val (conversationSection, _) = buildConversationSection(conversationHistory)
        val keyPoints =
            mapResponses
                .flatMapIndexed { idx, result ->
                    parsePoints(result.answer).map { point ->
                        point.copy(description = "----Analyst ${idx + 1}----\nImportance Score: ${point.score}\n${point.description}")
                    }
                }.filter { it.score > 0 || allowGeneralKnowledge }
        val sorted = keyPoints.sortedByDescending { it.score }
        if (sorted.isEmpty() && !allowGeneralKnowledge) {
            callbacks.forEach { it.onReduceResponseStart("") }
            callbacks.forEach { it.onReduceResponseEnd(NO_DATA_ANSWER) }
            return QueryResult(
                answer = NO_DATA_ANSWER,
                context = emptyList(),
                contextRecords = emptyMap(),
                llmCalls = 0,
                promptTokens = 0,
                outputTokens = 0,
                llmCallsCategories = mapOf("reduce" to 0),
                promptTokensCategories = mapOf("reduce" to 0),
                outputTokensCategories = mapOf("reduce" to 0),
                contextText = "",
            )
        }
        val buffer = StringBuilder()
        var tokens = if (conversationSection.isBlank()) 0 else tokenCount(conversationSection) + tokenCount("\n\n")
        if (conversationSection.isNotBlank()) {
            buffer.append(conversationSection).append("\n\n")
        }
        for (point in sorted) {
            val text = point.description + "\n\n"
            val newTokens = tokenCount(text)
            if (tokens + newTokens > maxDataTokens) break
            buffer.append(text)
            tokens += newTokens
        }
        val contextText = buffer.toString().trimEnd()
        val reducePrompt =
            reduceSystemPrompt
                .replace("{report_data}", contextText)
                .replace("{response_type}", responseType)
                .replace("{max_length}", reduceMaxLength.toString())
                .let { prompt -> if (allowGeneralKnowledge) "$prompt\n$generalKnowledgeInstruction" else prompt }
                .let { base ->
                    if (reduceParams.jsonResponse) "$base\nReturn ONLY valid JSON per the schema above." else base
                }
        val promptTokens = tokenCount(reducePrompt)
        val fullPrompt = "$reducePrompt\n\nUser question: $question"
        callbacks.forEach { it.onReduceResponseStart(contextText) }
        val answerText = streamAnswer(fullPrompt)
        val outputTokens = tokenCount(answerText)
        callbacks.forEach { it.onReduceResponseEnd(answerText) }
        return QueryResult(
            answer = answerText,
            context = emptyList(),
            contextRecords = emptyMap(),
            llmCalls = 1,
            promptTokens = promptTokens,
            outputTokens = outputTokens,
            llmCallsCategories = mapOf("reduce" to 1),
            promptTokensCategories = mapOf("reduce" to promptTokens),
            outputTokensCategories = mapOf("reduce" to outputTokens),
            contextText = contextText,
        )
    }

    /**
     * Extracts point objects from a JSON response into a list of Point values.
     *
     * Parses the top-level JSON for a "points" array and converts each entry with a "description"
     * string and optional numeric "score" into a Point. Entries missing a description are skipped;
     * missing or non-numeric scores are treated as 0. If the response is not valid JSON or lacks a
     * usable "points" array, an empty list is returned.
     *
     * @return A list of parsed Point objects, or an empty list if parsing fails or no valid points are found.
     */
    private fun parsePoints(response: String): List<Point> {
        return runCatching {
            val element = Json.parseToJsonElement(response)
            val points = (element as? JsonObject)?.get("points") as? JsonArray ?: return emptyList()
            points.mapNotNull { item ->
                val obj = item as? JsonObject ?: return@mapNotNull null
                val desc = obj["description"]?.jsonPrimitive?.content ?: return@mapNotNull null
                val score = obj["score"]?.jsonPrimitive?.content?.toIntOrNull() ?: 0
                Point(description = desc, score = score)
            }
        }.getOrElse { emptyList() }
    }

    /**
     * Count the number of tokens in the given text using the configured encoding.
     *
     * @return The number of tokens in the given text.
     */
    private fun tokenCount(text: String): Int = encoding.countTokens(text)

    /**
     * Builds a textual conversation section and corresponding context records from a list of history turns.
     *
     * Each turn is formatted as "turn|content" in a block prefixed by "-----Conversation-----" and a header row.
     *
     * @param history Ordered list of conversation turns (earliest first).
     * @return A Pair where the first element is the formatted conversation section string and the second element is a list of mutable maps for each turn with keys:
     * - `"turn"`: turn index (1-based),
     * - `"content"`: turn text,
     * - `"in_context"`: `"true"`.
     */
    private fun buildConversationSection(history: List<String>): Pair<String, List<MutableMap<String, String>>> {
        if (history.isEmpty()) return "" to emptyList()
        val header = "turn|content"
        val rows = history.mapIndexed { idx, turn -> "${idx + 1}|$turn" }
        val section = "-----Conversation-----\n$header\n${rows.joinToString("\n")}"
        val records =
            rows.mapIndexed { idx, row ->
                val parts = row.split("|", limit = 2)
                mutableMapOf(
                    "turn" to parts.getOrElse(0) { (idx + 1).toString() },
                    "content" to parts.getOrElse(1) { "" },
                    "in_context" to "true",
                )
            }
        return section to records
    }

    /**
     * Builds a single text block containing the question followed by conversation turns, each on its own line.
     *
     * @param question The user's current question (placed first).
     * @param history Prior conversation turns; each entry is appended as a separate line after the question.
     * @return The concatenated text: the question followed by each history entry on its own line, with no trailing newline.
     */
    private fun appendConversationHistory(
        question: String,
        history: List<String>,
    ): String =
        if (history.isEmpty()) {
            question
        } else {
            buildString {
                appendLine(question)
                history.forEach { appendLine(it) }
            }.trimEnd()
        }

    /**
     * Sends a prompt to the streaming model, collects streamed tokens, and returns the full response.
     *
     * Streams partial tokens to registered callbacks as they arrive and blocks until the complete
     * response is available (up to 5 minutes).
     *
     * @param prompt The text prompt to send to the streaming model.
     * @return The concatenated full response produced by the streaming model.
     */
    private fun streamAnswer(prompt: String): String {
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
        return future.get(5, TimeUnit.MINUTES)
    }

    /**
     * Streams the reduce-phase answer for the given question as incremental text chunks produced by the LLM.
     *
     * Builds context from the provided conversation history and selected reports, runs the map phase over context
     * chunks, extracts and composes key points, and then streams the reduce prompt's response tokens as they arrive.
     *
     * @param question The user's question to answer.
     * @param conversationHistory Ordered list of prior conversation turns to include in the context (optional).
     * @return A Flow<String> that emits partial response strings representing incremental LLM output; the complete answer
     *         is the concatenation of all emitted chunks, and the flow completes when the reduce response finishes.
     */
    fun streamSearch(
        question: String,
        conversationHistory: List<String> = emptyList(),
    ): Flow<String> =
        callbackFlow {
            val contextResult = buildContextChunks(question, conversationHistory)
            callbacks.forEach { it.onContext(contextResult.contextRecords) }
            callbacks.forEach { it.onMapResponseStart(contextResult.chunks) }

            val mapResponses =
                coroutineScope {
                    contextResult.chunks.map { chunk -> async { mapStep(question, chunk) } }.awaitAll()
                }
            callbacks.forEach { it.onMapResponseEnd(mapResponses) }

            val keyPoints =
                mapResponses
                    .flatMapIndexed { idx, result ->
                        parsePoints(result.answer).map { point ->
                            point.copy(description = "----Analyst ${idx + 1}----\nImportance Score: ${point.score}\n${point.description}")
                        }
                    }.filter { it.score > 0 || allowGeneralKnowledge }
            val sorted = keyPoints.sortedByDescending { it.score }
            if (sorted.isEmpty() && !allowGeneralKnowledge) {
                callbacks.forEach { it.onReduceResponseStart("") }
                callbacks.forEach { it.onReduceResponseEnd(NO_DATA_ANSWER) }
                trySend(NO_DATA_ANSWER)
                close()
                return@callbackFlow
            }
            val (conversationSection, _) = buildConversationSection(conversationHistory)
            val buffer = StringBuilder()
            var tokens = if (conversationSection.isBlank()) 0 else tokenCount(conversationSection) + tokenCount("\n\n")
            if (conversationSection.isNotBlank()) {
                buffer.append(conversationSection).append("\n\n")
            }
            for (point in sorted) {
                val text = point.description + "\n\n"
                val newTokens = tokenCount(text)
                if (tokens + newTokens > maxDataTokens) break
                buffer.append(text)
                tokens += newTokens
            }
            val contextText = buffer.toString().trimEnd()
            val reducePrompt =
                reduceSystemPrompt
                    .replace("{report_data}", contextText)
                    .replace("{response_type}", responseType)
                    .replace("{max_length}", reduceMaxLength.toString())
                    .let { prompt ->
                        if (allowGeneralKnowledge) "$prompt\n$generalKnowledgeInstruction" else prompt
                    }.let { base ->
                        if (reduceParams.jsonResponse) "$base\nReturn ONLY valid JSON per the schema above." else base
                    }
            callbacks.forEach { it.onReduceResponseStart(contextText) }
            val fullPrompt = "$reducePrompt\n\nUser question: $question"
            val builder = StringBuilder()
            streamingModel.chat(
                fullPrompt,
                object : StreamingChatResponseHandler {
                    override fun onPartialResponse(partialResponse: String) {
                        builder.append(partialResponse)
                        callbacks.forEach { it.onLLMNewToken(partialResponse) }
                        trySend(partialResponse)
                    }

                    override fun onCompleteResponse(response: ChatResponse) {
                        callbacks.forEach { it.onReduceResponseEnd(builder.toString()) }
                        close()
                    }

                    override fun onError(error: Throwable) {
                        close(error)
                    }
                },
            )
            awaitClose {}
        }

    private data class Point(
        val description: String,
        val score: Int,
    )

    companion object {
        private const val NO_DATA_ANSWER = "I do not know."

        internal val DEFAULT_COMMUNITY_RATING_PROMPT =
            """
            ---Role---
            You are a helpful assistant responsible for deciding whether the provided information is useful in answering a given question, even if it is only partially relevant.
            ---Goal---
            On a scale from 0 to 5, please rate how relevant or helpful is the provided information in answering the question.
            ---Information---
            {description}
            ---Question---
            {question}
            ---Target response length and format---
            Please response in the following JSON format with two entries:
            - "reason": the reasoning of your rating, please include information that you have considered.
            - "rating": the relevancy rating from 0 to 5, where 0 is the least relevant and 5 is the most relevant.
            {
                "reason": str,
                "rating": int.
            }
            """.trimIndent()

        internal val DEFAULT_MAP_SYSTEM_PROMPT: String =
            """
            ---Role---

            You are a helpful assistant responding to questions about data in the tables provided.


            ---Goal---

            Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

            You should use the data provided in the data tables below as the primary context for generating the response.
            If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

            Each key point in the response should have the following element:
            - Description: A comprehensive description of the point.
            - Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

            The response should be JSON formatted as follows:
            {{
                "points": [
                    {{"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}},
                    {{"description": "Description of point 2 [Data: Reports (report ids)]", "score": score_value}}
                ]
            }}

            The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

            Points supported by data should list the relevant reports as references as follows:
            "This is an example sentence supported by data references [Data: Reports (report ids)]"

            **Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

            For example:
            "Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

            where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data report in the provided tables.

            Do not include information where the supporting evidence for it is not provided.

            Limit your response length to {max_length} words.

            ---Data tables---

            {context_data}

            """.trimIndent()

        internal val DEFAULT_REDUCE_SYSTEM_PROMPT: String =
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

        internal val DEFAULT_GENERAL_KNOWLEDGE_INSTRUCTION: String =
            """
            The response may also include relevant real-world knowledge outside the dataset, but it must be explicitly annotated with a verification tag [LLM: verify]. For example:
            "This is an example sentence supported by real-world knowledge [LLM: verify]."
            """.trimIndent()
    }
}
