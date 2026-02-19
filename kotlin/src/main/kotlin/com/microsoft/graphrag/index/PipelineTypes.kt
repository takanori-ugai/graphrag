package com.microsoft.graphrag.index

import com.microsoft.graphrag.logger.Progress
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import java.nio.file.Path

/**
 * Function signature for workflow steps in a pipeline.
 */
typealias WorkflowFunction = suspend (config: GraphRagConfig, context: PipelineRunContext) -> WorkflowResult

/**
 * Result returned by a workflow step.
 *
 * @property result Arbitrary workflow result payload.
 * @property stop Whether to stop subsequent workflows.
 */
data class WorkflowResult(
    val result: Any? = null,
    val stop: Boolean = false,
)

/**
 * Result emitted after a workflow run.
 *
 * @property workflow Workflow name.
 * @property result Workflow result payload.
 * @property state Mutable pipeline state after the workflow.
 * @property errors Errors collected during the workflow, if any.
 */
data class PipelineRunResult(
    val workflow: String,
    val result: Any?,
    val state: MutableMap<String, Any?>,
    val errors: List<String>?,
)

/**
 * Accumulated runtime statistics for a pipeline run.
 *
 * @property workflows Workflow-specific runtime metrics.
 * @property totalRuntime Total runtime in seconds.
 */
data class PipelineRunStats(
    val workflows: MutableMap<String, Map<String, Double>> = linkedMapOf(),
    var totalRuntime: Double = 0.0,
)

/**
 * Resolved filesystem paths used by the pipeline.
 *
 * @property rootDir Project root directory.
 * @property inputDir Input directory containing source documents.
 * @property outputDir Output directory for index artifacts.
 * @property updateOutputDir Output directory for update artifacts.
 */
data class GraphRagConfig(
    val rootDir: Path,
    val inputDir: Path,
    val outputDir: Path,
    val updateOutputDir: Path,
)

/**
 * Callbacks invoked during workflow execution.
 */
interface WorkflowCallbacks {
    /**
     * Called when a workflow starts.
     *
     * @param name Workflow name.
     */
    fun workflowStart(name: String)

    /**
     * Called when a workflow finishes to notify listeners of its outcome.
     *
     * @param name The workflow name.
     * @param result The workflow's outcome, containing the resulting value (nullable) and a `stop` flag
     * indicating whether subsequent workflows should be halted.
     */
    fun workflowEnd(
        name: String,
        result: WorkflowResult,
    )

    /**
     * Receives progress updates during workflow execution.
     *
     * The default implementation performs no action; implementations may override to report or react to
     * progress (for example, percentage complete, messages, or stage information).
     *
     * @param progress Progress information reported by a workflow.
     */
    fun progress(progress: Progress) {
        // default no-op
    }
}

/**
 * No-op WorkflowCallbacks implementation.
 */
class NoopWorkflowCallbacks : WorkflowCallbacks {
    override fun workflowStart(name: String) {
        // no-op
    }

    /**
     * No-op implementation invoked when a workflow completes.
     *
     * @param name The workflow's name.
     * @param result The workflow's result and stop flag.
     */
    override fun workflowEnd(
        name: String,
        result: WorkflowResult,
    ) {
        // no-op
    }

    /**
     * Ignores a progress update without performing any action.
     *
     * @param progress The progress event to ignore.
     */
    override fun progress(progress: Progress) {
        // no-op
    }
}

/**
 * Pipeline of named workflow steps.
 */
interface Pipeline {
    /**
     * Returns the workflow sequence to execute.
     *
     * @return Sequence of workflow name and function pairs.
     */
    fun run(): Sequence<Pair<String, WorkflowFunction>>

    /**
     * Removes a workflow by name.
     *
     * @param name Workflow name to remove.
     */
    fun remove(name: String)
}

/**
 * Simple in-memory pipeline implementation.
 */
class SimplePipeline(
    workflows: List<Pair<String, WorkflowFunction>>,
) : Pipeline {
    private val items: MutableList<Pair<String, WorkflowFunction>> = workflows.toMutableList()

    override fun run(): Sequence<Pair<String, WorkflowFunction>> = items.asSequence()

    override fun remove(name: String) {
        items.removeIf { it.first == name }
    }
}

/**
 * Runtime context passed to workflow steps.
 *
 * @property inputStorage Storage for input artifacts.
 * @property outputStorage Storage for output artifacts.
 * @property cache Cache used by workflows.
 * @property callbacks Workflow callbacks.
 * @property state Mutable pipeline state map.
 * @property stats Runtime statistics accumulator.
 * @property previousStorage Optional storage for previous runs.
 */
data class PipelineRunContext(
    val inputStorage: PipelineStorage,
    val outputStorage: PipelineStorage,
    val cache: PipelineCache,
    val callbacks: WorkflowCallbacks,
    val state: MutableMap<String, Any?> = linkedMapOf(),
    val stats: PipelineRunStats = PipelineRunStats(),
    val previousStorage: PipelineStorage? = null,
)

/**
 * Cache abstraction for pipeline steps.
 */
interface PipelineCache {
    /**
     * Reads a cached value.
     *
     * @param key Cache key.
     * @return Cached content or null when missing.
     */
    suspend fun get(key: String): String?

    /**
     * Writes a cached value.
     *
     * @param key Cache key.
     * @param value Content to cache.
     */
    suspend fun set(
        key: String,
        value: String,
    )
}

/**
 * No-op cache implementation.
 */
class NoopCache : PipelineCache {
    override suspend fun get(key: String): String? = null

    override suspend fun set(
        key: String,
        value: String,
    ) {
        // no-op
    }
}

/**
 * Builds a Flow from a callback-based workflow result builder.
 *
 * @param builder Function that emits PipelineRunResult items via a provided emitter.
 * @return A Flow that emits PipelineRunResult items.
 */
fun flowFromResults(builder: suspend (suspend (PipelineRunResult) -> Unit) -> Unit): Flow<PipelineRunResult> =
    flow {
        builder { emit(it) }
    }
