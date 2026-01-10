package com.microsoft.graphrag.index

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import java.nio.file.Path

typealias WorkflowFunction = suspend (config: GraphRagConfig, context: PipelineRunContext) -> WorkflowResult

data class WorkflowResult(
    val result: Any? = null,
    val stop: Boolean = false,
)

data class PipelineRunResult(
    val workflow: String,
    val result: Any?,
    val state: MutableMap<String, Any?>,
    val errors: List<String>?,
)

data class PipelineRunStats(
    val workflows: MutableMap<String, Map<String, Double>> = linkedMapOf(),
    var totalRuntime: Double = 0.0,
)

data class GraphRagConfig(
    val rootDir: Path,
    val inputDir: Path,
    val outputDir: Path,
    val updateOutputDir: Path,
)

interface WorkflowCallbacks {
    fun workflowStart(name: String)

    fun workflowEnd(
        name: String,
        result: WorkflowResult,
    )
}

class NoopWorkflowCallbacks : WorkflowCallbacks {
    override fun workflowStart(name: String) {
        // no-op
    }

    override fun workflowEnd(
        name: String,
        result: WorkflowResult,
    ) {
        // no-op
    }
}

interface Pipeline {
    fun run(): Sequence<Pair<String, WorkflowFunction>>

    fun remove(name: String)
}

class SimplePipeline(
    workflows: List<Pair<String, WorkflowFunction>>,
) : Pipeline {
    private val items: MutableList<Pair<String, WorkflowFunction>> = workflows.toMutableList()

    override fun run(): Sequence<Pair<String, WorkflowFunction>> = items.asSequence()

    override fun remove(name: String) {
        items.removeIf { it.first == name }
    }
}

data class PipelineRunContext(
    val inputStorage: PipelineStorage,
    val outputStorage: PipelineStorage,
    val cache: PipelineCache,
    val callbacks: WorkflowCallbacks,
    val state: MutableMap<String, Any?> = linkedMapOf(),
    val stats: PipelineRunStats = PipelineRunStats(),
    val previousStorage: PipelineStorage? = null,
)

interface PipelineCache {
    suspend fun get(key: String): String?

    suspend fun set(
        key: String,
        value: String,
    )
}

class NoopCache : PipelineCache {
    override suspend fun get(key: String): String? = null

    override suspend fun set(
        key: String,
        value: String,
    ) {
        // no-op
    }
}

fun <T> flowFromResults(builder: suspend (suspend (PipelineRunResult) -> Unit) -> Unit): Flow<PipelineRunResult> =
    flow {
        builder { emit(it) }
    }
