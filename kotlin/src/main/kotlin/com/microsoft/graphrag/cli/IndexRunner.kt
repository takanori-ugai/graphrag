package com.microsoft.graphrag.cli

import com.microsoft.graphrag.index.GraphRagConfig
import com.microsoft.graphrag.index.NoopWorkflowCallbacks
import com.microsoft.graphrag.index.PipelineRunResult
import com.microsoft.graphrag.index.defaultPipeline
import com.microsoft.graphrag.index.runPipeline
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.runBlocking
import java.io.PrintStream
import java.nio.file.Path

data class IndexOptions(
    val root: Path,
    val config: Path?,
    val method: String,
    val verbose: Boolean,
    val memprofile: Boolean,
    val dryRun: Boolean,
    val cache: Boolean,
    val skipValidation: Boolean,
    val output: Path?,
    val isUpdate: Boolean,
)

class IndexRunner(
    private val out: PrintStream = System.out,
) {
    fun run(options: IndexOptions) {
        out.println("Index command received (update=${options.isUpdate})")
        printOptions(options)

        if (options.dryRun) {
            out.println("Dry run requested: validating configuration only (placeholder).")
            return
        }

        val config =
            GraphRagConfig(
                rootDir = options.root,
                inputDir = options.root.resolve("input"),
                outputDir = options.output ?: options.root.resolve("output"),
                updateOutputDir = options.output ?: options.root.resolve("update_output"),
            )

        runBlocking {
            runPipeline(
                pipeline = defaultPipeline(),
                config = config,
                callbacks = NoopWorkflowCallbacks(),
                isUpdateRun = options.isUpdate,
            ).collect { result: PipelineRunResult ->
                out.println("Workflow ${result.workflow} completed (errors=${result.errors?.size ?: 0})")
            }
        }

        out.println("Index pipeline completed (placeholder).")
    }

    private fun printOptions(options: IndexOptions) {
        out.println("root=${options.root}")
        out.println("config=${options.config}")
        out.println("method=${options.method}")
        out.println("output=${options.output}")
        out.println("verbose=${options.verbose}")
        out.println("memprofile=${options.memprofile}")
        out.println("dryRun=${options.dryRun}")
        out.println("cache=${options.cache}")
        out.println("skipValidation=${options.skipValidation}")
    }
}
