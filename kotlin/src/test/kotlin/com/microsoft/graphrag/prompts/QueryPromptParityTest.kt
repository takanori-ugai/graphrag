package com.microsoft.graphrag.prompts

import com.microsoft.graphrag.prompts.query.QUESTION_SYSTEM_PROMPT
import kotlin.test.Test
import kotlin.test.assertTrue

class QueryPromptParityTest {
    @Test
    fun `query prompt resources use json output`() {
        val cases =
            listOf(
                PromptCase(
                    resourcePath = "prompts/query/basic_search_system_prompt.txt",
                    requiredFragments = listOf("\"response\"", "\"score\"", "\"follow_up_queries\""),
                ),
                PromptCase(
                    resourcePath = "prompts/query/local_search_system_prompt.txt",
                    requiredFragments = listOf("\"response\"", "\"score\"", "\"follow_up_queries\""),
                ),
                PromptCase(
                    resourcePath = "prompts/query/global_search_map_system_prompt.txt",
                    requiredFragments = listOf("\"points\""),
                ),
                PromptCase(
                    resourcePath = "prompts/query/global_search_reduce_system_prompt.txt",
                    requiredFragments = listOf("\"response\"", "\"score\"", "\"follow_up_queries\""),
                ),
                PromptCase(
                    resourcePath = "prompts/query/drift_search_system_prompt.txt",
                    requiredFragments = listOf("\"response\"", "\"score\"", "\"follow_up_queries\""),
                ),
                PromptCase(
                    resourcePath = "prompts/query/drift_search_reduce_system_prompt.txt",
                    requiredFragments = listOf("\"response\"", "\"score\"", "\"follow_up_queries\""),
                ),
                PromptCase(
                    resourcePath = "prompts/query/global_search_general_knowledge_prompt.txt",
                    requiredFragments = emptyList(),
                ),
            )

        cases.forEach { case ->
            val kotlinPrompt = loadResource(case.resourcePath)
            case.requiredFragments.forEach { fragment ->
                assertTrue(
                    kotlinPrompt.contains(fragment),
                    "Prompt ${case.resourcePath} missing expected fragment: $fragment",
                )
            }
        }
    }

    @Test
    fun `question generation prompt requests json`() {
        assertTrue(
            QUESTION_SYSTEM_PROMPT.contains("\"questions\""),
            "Question generation prompt missing JSON questions schema",
        )
    }

    private data class PromptCase(
        val resourcePath: String,
        val requiredFragments: List<String>,
    )

    private fun loadResource(path: String): String {
        return (this::class.java.classLoader.getResourceAsStream(path)
            ?: error("Resource not found: $path"))
            .bufferedReader(Charsets.UTF_8)
            .use { it.readText() }
    }

}
