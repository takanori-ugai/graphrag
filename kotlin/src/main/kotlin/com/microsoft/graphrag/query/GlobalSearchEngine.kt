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

/**
 * Global search engine that mirrors Python's map-reduce flow over community reports.
 */
class GlobalSearchEngine(
    private val streamingModel: OpenAiStreamingChatModel,
    private val communityReports: List<CommunityReport>,
    private val callbacks: List<QueryCallbacks> = emptyList(),
    private val mapSystemPrompt: String = MAP_SYSTEM_PROMPT,
    private val reduceSystemPrompt: String = REDUCE_SYSTEM_PROMPT,
    private val responseType: String = "multiple paragraphs",
    private val allowGeneralKnowledge: Boolean = false,
    private val generalKnowledgeInstruction: String = GENERAL_KNOWLEDGE_INSTRUCTION,
    private val mapMaxLength: Int = 1_000,
    private val reduceMaxLength: Int = 2_000,
    private val maxContextTokens: Int = 8_000,
    private val encoding: Encoding = Encodings.newLazyEncodingRegistry().getEncoding(EncodingType.CL100K_BASE),
) {
    suspend fun search(question: String): GlobalSearchResult {
        val contextChunks = buildContextChunks()
        callbacks.forEach { it.onContext(mapOf("reports" to contextChunks.map { mutableMapOf("in_context" to "true") })) }
        callbacks.forEach { it.onMapResponseStart(contextChunks) }

        val mapResponses = contextChunks.map { chunk -> mapStep(question, chunk) }
        callbacks.forEach { it.onMapResponseEnd(mapResponses) }

        val reduceResult = reduceStep(question, mapResponses)

        val llmCallsCategories =
            mapOf(
                "build_context" to 0,
                "map" to mapResponses.sumOf { it.llmCalls },
                "reduce" to reduceResult.llmCalls,
            )
        val promptTokensCategories =
            mapOf(
                "build_context" to 0,
                "map" to mapResponses.sumOf { it.promptTokens },
                "reduce" to reduceResult.promptTokens,
            )
        val outputTokensCategories =
            mapOf(
                "build_context" to 0,
                "map" to mapResponses.sumOf { it.outputTokens },
                "reduce" to reduceResult.outputTokens,
            )

        return GlobalSearchResult(
            answer = reduceResult.answer,
            mapResponses = mapResponses,
            reduceContextText = reduceResult.contextText,
            contextRecords = mapOf("reports" to contextChunks.map { mutableMapOf("in_context" to "true") }),
            llmCalls = llmCallsCategories.values.sum(),
            promptTokens = promptTokensCategories.values.sum(),
            outputTokens = outputTokensCategories.values.sum(),
            llmCallsCategories = llmCallsCategories,
            promptTokensCategories = promptTokensCategories,
            outputTokensCategories = outputTokensCategories,
        )
    }

    private fun buildContextChunks(): List<String> {
        if (communityReports.isEmpty()) return emptyList()
        val header = listOf("id", "title", "summary", "rank").joinToString("|")
        val sorted = communityReports.sortedByDescending { it.rank ?: 0.0 }

        val chunks = mutableListOf<String>()
        var current = StringBuilder("-----Reports-----\n$header\n")
        var tokens = tokenCount(current.toString())
        for (report in sorted) {
            val row =
                listOf(
                    report.id ?: report.communityId.toString(),
                    report.title ?: report.communityId.toString(),
                    report.summary,
                    (report.rank ?: 0.0).toString(),
                ).joinToString("|") + "\n"
            val rowTokens = tokenCount(row)
            if (tokens + rowTokens > maxContextTokens && current.isNotEmpty()) {
                chunks += current.toString().trimEnd()
                current = StringBuilder("-----Reports-----\n$header\n")
                tokens = tokenCount(current.toString())
            }
            current.append(row)
            tokens += rowTokens
        }
        if (current.isNotEmpty()) chunks += current.toString().trimEnd()
        return chunks
    }

    private suspend fun mapStep(
        question: String,
        context: String,
    ): QueryResult {
        val prompt =
            mapSystemPrompt
                .replace("{context_data}", context)
                .replace("{max_length}", mapMaxLength.toString())
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
    ): QueryResult {
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
        var tokens = 0
        for (point in sorted) {
            val text = point.description + "\n\n"
            val newTokens = tokenCount(text)
            if (tokens + newTokens > maxContextTokens) break
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

    fun streamSearch(question: String): Flow<String> =
        callbackFlow {
            val contextChunks = buildContextChunks()
            callbacks.forEach { it.onContext(mapOf("reports" to contextChunks.map { mutableMapOf("in_context" to "true") })) }
            callbacks.forEach { it.onMapResponseStart(contextChunks) }

            val mapResponses = contextChunks.map { chunk -> mapStep(question, chunk) }
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
                trySend(NO_DATA_ANSWER)
                close()
                return@callbackFlow
            }
            val buffer = StringBuilder()
            var tokens = 0
            for (point in sorted) {
                val text = point.description + "\n\n"
                val newTokens = tokenCount(text)
                if (tokens + newTokens > maxContextTokens) break
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

        private val MAP_SYSTEM_PROMPT: String =
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

        private val REDUCE_SYSTEM_PROMPT: String =
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

        private val GENERAL_KNOWLEDGE_INSTRUCTION: String =
            """
            The response may also include relevant real-world knowledge outside the dataset, but it must be explicitly annotated with a verification tag [LLM: verify]. For example:
            "This is an example sentence supported by real-world knowledge [LLM: verify]."
            """.trimIndent()
    }
}
