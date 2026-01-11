package com.microsoft.graphrag.query

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
    private val globalSearchEngine: GlobalSearchEngine,
    private val localQueryEngine: LocalQueryEngine,
) {
    suspend fun search(
        question: String,
        followUpQueries: List<String> = emptyList(),
    ): DriftSearchResult {
        var totalLlmCalls = 0
        var totalPromptTokens = 0
        var totalOutputTokens = 0

        val actions = mutableListOf<QueryResult>()

        // Primer: global search
        val primer = globalSearchEngine.search(question)
        totalLlmCalls += primer.llmCalls
        totalPromptTokens += primer.promptTokens
        totalOutputTokens += primer.outputTokens
        actions +=
            QueryResult(
                answer = primer.answer,
                context = emptyList(),
                contextRecords = primer.contextRecords,
                contextText = primer.reduceContextText,
                llmCalls = primer.llmCalls,
                promptTokens = primer.promptTokens,
                outputTokens = primer.outputTokens,
                llmCallsCategories = primer.llmCallsCategories,
                promptTokensCategories = primer.promptTokensCategories,
                outputTokensCategories = primer.outputTokensCategories,
            )

        // Local follow-ups (simulate DRIFT actions)
        val followUps = if (followUpQueries.isEmpty()) listOf(question) else followUpQueries
        followUps.forEach { followUp ->
            val local = localQueryEngine.answer(followUp, responseType = "multiple paragraphs")
            totalLlmCalls += 1 // local answer call
            totalPromptTokens += 0 // not tracked in LocalQueryEngine
            totalOutputTokens += 0
            actions +=
                QueryResult(
                    answer = local.answer,
                    context = local.context,
                    contextRecords = local.contextRecords,
                    contextText = "",
                    llmCalls = 1,
                    promptTokens = 0,
                    outputTokens = 0,
                )
        }

        val categories =
            mapOf(
                "primer" to actions.firstOrNull()?.llmCalls.orZero(),
                "local" to actions.drop(1).sumOf { it.llmCalls },
            )

        return DriftSearchResult(
            answer = actions.lastOrNull()?.answer ?: "",
            actions = actions,
            llmCalls = totalLlmCalls,
            promptTokens = totalPromptTokens,
            outputTokens = totalOutputTokens,
            llmCallsCategories = categories,
            promptTokensCategories = mapOf("primer" to totalPromptTokens, "local" to 0),
            outputTokensCategories = mapOf("primer" to totalOutputTokens, "local" to 0),
        )
    }

    private fun Int?.orZero(): Int = this ?: 0
}
