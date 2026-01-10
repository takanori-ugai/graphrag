package com.microsoft.graphrag

import com.microsoft.graphrag.index.GraphRagConfig
import com.microsoft.graphrag.index.NoopWorkflowCallbacks
import com.microsoft.graphrag.index.defaultPipeline
import com.microsoft.graphrag.index.runPipeline
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
        GraphRAG is a retrieval system that builds a knowledge graph from text.
        It extracts entities, relationships, and claims, then writes parquet outputs.
        """.trimIndent()
    Files.writeString(inputDir.resolve("doc1.txt"), sampleText)

    val config = GraphRagConfig(root, inputDir, outputDir, updateDir)
    val pipeline = defaultPipeline()
    val callbacks = NoopWorkflowCallbacks()

    runBlocking {
        runPipeline(pipeline, config, callbacks).collect { result ->
            println("Workflow '${result.workflow}' completed; errors=${result.errors?.joinToString()}")
        }
    }

    println("Indexing complete. Outputs in $outputDir")
}
