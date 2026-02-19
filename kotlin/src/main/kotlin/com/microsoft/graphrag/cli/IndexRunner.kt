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

/**
 * CLI options used to configure an indexing run.
 *
 * @property root Project root directory.
 * @property config Optional settings file path.
 * @property method Indexing method identifier.
 * @property verbose Whether to enable verbose logging.
 * @property memprofile Whether to enable memory profiling.
 * @property dryRun Whether to validate configuration without executing steps.
 * @property cache Whether LLM caching is enabled.
 * @property skipValidation Whether to skip preflight validation.
 * @property output Optional output directory override.
 * @property isUpdate Whether this run updates an existing index.
 */
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

/**
 * Executes the indexing pipeline based on CLI options.
 *
 * @property out Output stream used for status reporting.
 */
class IndexRunner(
    private val out: PrintStream = System.out,
) {
    /**
     * Runs the indexing workflow using the provided options.
     *
     * @param options Options describing the indexing run.
     */
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
                outputDir = options.output?.resolve("output") ?: options.root.resolve("output"),
                updateOutputDir = options.output?.resolve("update_output") ?: options.root.resolve("update_output"),
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
