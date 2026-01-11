package com.microsoft.graphrag.query

import com.microsoft.graphrag.index.CommunityReport
import com.microsoft.graphrag.index.Entity
import com.microsoft.graphrag.index.EntitySummary
import com.microsoft.graphrag.index.LocalVectorStore
import com.microsoft.graphrag.index.TextEmbedding
import com.microsoft.graphrag.index.TextUnit
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.model.output.Response
import dev.langchain4j.service.AiServices
import dev.langchain4j.service.SystemMessage
import dev.langchain4j.service.UserMessage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.sqrt

/**
 * Local search mirrors the Python local search by prioritizing entity-centric
 * context (entity summaries if available, otherwise source chunks). It falls back
 * to text-unit similarity if no entity vectors are present.
 */
@Suppress("LongParameterList")
class LocalQueryEngine(
    private val chatModel: OpenAiChatModel,
    private val embeddingModel: EmbeddingModel,
    private val vectorStore: LocalVectorStore,
    private val textUnits: List<TextUnit>,
    private val textEmbeddings: List<TextEmbedding>,
    private val entities: List<Entity>,
    private val entitySummaries: List<EntitySummary>,
    private val topKEntities: Int = 5,
    private val fallbackTopK: Int = 5,
    private val maxContextChars: Int = 800,
) {
    suspend fun answer(
        question: String,
        responseType: String,
    ): QueryResult {
        val context = buildContext(question)
        val prompt = buildPrompt(responseType, context)
        val answer = generate(prompt)
        return QueryResult(answer = answer, context = context)
    }

    @Suppress("ReturnCount")
    suspend fun buildContext(question: String): List<QueryContextChunk> {
        val queryEmbedding = embed(question) ?: return emptyList()
        val entityContexts = selectEntityContext(queryEmbedding)
        if (entityContexts.isNotEmpty()) return entityContexts
        return selectTextContext(queryEmbedding)
    }

    private fun selectEntityContext(queryEmbedding: List<Double>): List<QueryContextChunk> {
        val summariesById = entitySummaries.associateBy { it.entityId }
        val entitiesById = entities.associateBy { it.id }
        return vectorStore
            .nearestEntities(queryEmbedding, topKEntities)
            .mapNotNull { (entityId, distance) ->
                val text =
                    summariesById[entityId]?.summary
                        ?: entitiesById[entityId]?.let { entity ->
                            textUnits.find { it.chunkId == entity.sourceChunkId }?.text
                        }
                text?.let { QueryContextChunk(id = entityId, text = it, score = distance) }
            }
    }

    private fun selectTextContext(queryEmbedding: List<Double>): List<QueryContextChunk> {
        val byChunkId = textUnits.associateBy { it.chunkId }
        return textEmbeddings
            .mapNotNull { embedding ->
                val textUnit = byChunkId[embedding.chunkId] ?: return@mapNotNull null
                val score = cosineSimilarity(queryEmbedding, embedding.vector)
                QueryContextChunk(id = textUnit.id, text = textUnit.text, score = score)
            }.sortedByDescending { it.score }
            .take(fallbackTopK)
    }

    private fun buildPrompt(
        responseType: String,
        context: List<QueryContextChunk>,
    ): String {
        val header = "source_id|text"
        val rows =
            context.joinToString("\n") { chunk ->
                "${chunk.id}|${chunk.text.take(maxContextChars)}"
            }
        val contextBlock = "$header\n$rows"
        return LOCAL_SEARCH_SYSTEM_PROMPT
            .replace("{context_data}", contextBlock)
            .replace("{response_type}", responseType)
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
 * Global search leans on community reports; it embeds the summaries and selects
 * the most relevant communities as context.
 */
class GlobalQueryEngine(
    private val chatModel: OpenAiChatModel,
    private val embeddingModel: EmbeddingModel,
    private val communityReports: List<CommunityReport>,
    private val topK: Int = 3,
    private val maxContextChars: Int = 1000,
) {
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
                    val summaryEmbedding = embed(report.summary) ?: return@mapNotNull null
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
