package com.microsoft.graphrag

import com.microsoft.graphrag.index.GraphRagConfig
import com.microsoft.graphrag.index.WorkflowCallbacks
import com.microsoft.graphrag.index.WorkflowResult
import com.microsoft.graphrag.index.defaultPipeline
import com.microsoft.graphrag.index.runPipeline
import com.microsoft.graphrag.logger.Progress
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.runBlocking
import java.nio.file.Files
import java.nio.file.Path

/**
 * Minimal example that indexes a single text file using the default pipeline.
 *
 * Usage:
 * ```
 * OPENAI_API_KEY=... ./gradlew :kotlin:run --args="sample-index"
 * ```
 */
fun main() {
    val apiKey = System.getenv("OPENAI_API_KEY")
    if (apiKey.isNullOrBlank()) {
        println("OPENAI_API_KEY is not set. Please set it to run the indexer.")
        return
    }
    val root = Path.of("sample-index")
    val inputDir = root.resolve("input")
    val outputDir = root.resolve("output")
    val updateDir = root.resolve("update")
    Files.createDirectories(inputDir)
    Files.createDirectories(outputDir)
    Files.createDirectories(updateDir)

    // Write a toy document to index.
    val sampleText =
        """
        GraphRAG is a retrieval system that builds a knowledge graph from text. It extracts entities like Microsoft and
        OpenAI, then links them through relationships such as partnership and research collaboration. In 2023, Microsoft
        invested in OpenAI to accelerate AI research. OpenAI builds large language models and deploys them via Azure.
        Azure provides GPU infrastructure, and GitHub Copilot is powered by OpenAI models. Satya Nadella leads Microsoft,
        while Sam Altman leads OpenAI. Together, the companies publish research papers, improve safety, and scale
        responsible AI deployments across products like Bing Chat and GitHub Copilot.
        """.trimIndent()
    Files.writeString(inputDir.resolve("doc1.txt"), sampleText)

    val config = GraphRagConfig(root, inputDir, outputDir, updateDir)
    val pipeline = defaultPipeline()
    val callbacks = LoggingCallbacks()

    runBlocking {
        runPipeline(pipeline, config, callbacks).collect { result ->
            println("Workflow '${result.workflow}' completed; errors=${result.errors?.joinToString()}")
        }
    }

    println("Indexing complete. Outputs in $outputDir")
}

private class LoggingCallbacks : WorkflowCallbacks {
    override fun workflowStart(name: String) {
        logger.info { "Starting workflow: $name" }
    }

    override fun workflowEnd(
        name: String,
        result: WorkflowResult,
    ) {
        logger.info { "Finished workflow: $name (stop=${result.stop})" }
    }

    override fun progress(progress: Progress) {
        val desc = progress.description ?: "progress"
        val complete = progress.completedItems ?: 0
        val total = progress.totalItems ?: 0
        logger.info { "$desc$complete/$total" }
    }

    companion object {
        private val logger = KotlinLogging.logger {}
    }
}
