package com.microsoft.graphrag.index

import java.io.InputStreamReader
import java.nio.charset.StandardCharsets

/**
 * Loads prompt templates from bundled resources.
 */
class PromptRepository {
    /**
     * Loads the extract-graph prompt template.
     *
     * @return Prompt template text.
     */
    fun loadExtractGraphPrompt(): String = loadResource("prompts/index/extract_graph.txt")

    /**
     * Loads the community report prompt template.
     *
     * @return Prompt template text.
     */
    fun loadCommunityReportPrompt(): String = loadResource("prompts/index/community_report.txt")

    /**
     * Loads the summarize-descriptions prompt template.
     *
     * @return Prompt template text.
     */
    fun loadSummarizeDescriptionsPrompt(): String = loadResource("prompts/index/summarize_descriptions.txt")

    /**
     * Loads the extract-claims prompt template.
     *
     * @return Prompt template text.
     */
    fun loadExtractClaimsPrompt(): String = loadResource("prompts/index/extract_claims.txt")

    private fun loadResource(path: String): String {
        val stream =
            this::class.java.classLoader.getResourceAsStream(path)
                ?: error("Prompt resource not found: $path")
        return InputStreamReader(stream, StandardCharsets.UTF_8).readText()
    }
}
