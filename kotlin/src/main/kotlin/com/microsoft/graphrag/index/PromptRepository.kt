package com.microsoft.graphrag.index

import java.io.InputStreamReader
import java.nio.charset.StandardCharsets

class PromptRepository {
    fun loadExtractGraphPrompt(): String = loadResource("prompts/index/extract_graph.txt")

    fun loadCommunityReportPrompt(): String = loadResource("prompts/index/community_report.txt")

    fun loadSummarizeDescriptionsPrompt(): String = loadResource("prompts/index/summarize_descriptions.txt")

    fun loadExtractClaimsPrompt(): String = loadResource("prompts/index/extract_claims.txt")

    private fun loadResource(path: String): String {
        val stream =
            this::class.java.classLoader.getResourceAsStream(path)
                ?: error("Prompt resource not found: $path")
        return InputStreamReader(stream, StandardCharsets.UTF_8).readText()
    }
}
