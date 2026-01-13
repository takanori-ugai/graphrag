package com.microsoft.graphrag.query

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingType
import com.microsoft.graphrag.index.CommunityReport
import com.microsoft.graphrag.index.CommunityReportEmbedding
import dev.langchain4j.model.chat.response.ChatResponse
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler
import dev.langchain4j.model.embedding.EmbeddingModel
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
import kotlin.math.min

data class GlobalSearchResult(
    val answer: String,
    val mapResponses: List<QueryResult>,
    val reduceContextText: String,
    val contextRecords: Map<String, List<MutableMap<String, String>>>,
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
    communityReportEmbeddings: List<CommunityReportEmbedding> = emptyList(),
    private val embeddingModel: EmbeddingModel? = null,
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
                contextRecords = contextResult.contextRecords,
                llmCalls = llmCallsCategories.values.sum(),
                promptTokens = promptTokensCategories.values.sum(),
                outputTokens = outputTokensCategories.values.sum(),
                llmCallsCategories = llmCallsCategories,
                promptTokensCategories = promptTokensCategories,
                outputTokensCategories = outputTokensCategories,
            )
        }

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
            contextRecords = mapOf("reports" to listOf(mutableMapOf("in_context" to "true"))),
            llmCalls = 1,
            promptTokens = promptTokens,
            outputTokens = outputTokens,
            llmCallsCategories = mapOf("map" to 1),
            promptTokensCategories = mapOf("map" to promptTokens),
            outputTokensCategories = mapOf("map" to outputTokens),
        )
    }

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

    private fun tokenCount(text: String): Int = encoding.countTokens(text)

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
        return future.get()
    }

    fun streamSearch(
        question: String,
        conversationHistory: List<String> = emptyList(),
    ): Flow<String> =
        callbackFlow {
            val contextResult = buildContextChunks(question, conversationHistory)
            callbacks.forEach { it.onContext(contextResult.contextRecords) }
            callbacks.forEach { it.onMapResponseStart(contextResult.chunks) }

            val mapResponses =
                kotlinx.coroutines.runBlocking {
                    coroutineScope {
                        contextResult.chunks.map { chunk -> async { mapStep(question, chunk) } }.awaitAll()
                    }
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

            ---Goal---

            Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

            You should use the data provided in the data tables below as the primary context for generating the response.
            If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

            Each key point in the response should have the following element:
            - Description: A comprehensive description of the point.
            - Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

            The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

            Points supported by data should list the relevant reports as references as follows:
            "This is an example sentence supported by data references [Data: Reports (report ids)]"

            **Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

            For example:
            "Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

            where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data report in the provided tables.

            Do not include information where the supporting evidence for it is not provided.

            Limit your response length to {max_length} words.

            The response should be JSON formatted as follows:
            {{
                "points": [
                    {{"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}},
                    {{"description": "Description of point 2 [Data: Reports (report ids)]", "score": score_value}}
                ]
            }}
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
