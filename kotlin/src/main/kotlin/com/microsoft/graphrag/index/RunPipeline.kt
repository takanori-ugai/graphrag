package com.microsoft.graphrag.index

import com.microsoft.graphrag.logger.Progress
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.nio.file.Files
import java.nio.file.Path
import java.time.Instant

@Suppress("LongParameterList", "LongMethod", "TooGenericExceptionCaught")
suspend fun runPipeline(
    pipeline: Pipeline,
    config: GraphRagConfig,
    callbacks: WorkflowCallbacks,
    isUpdateRun: Boolean = false,
    additionalContext: Map<String, Any?>? = null,
    inputDocumentsJson: String? = null,
): Flow<PipelineRunResult> =
    flow {
        val inputStorage = FilePipelineStorage(config.inputDir)
        val outputStorage = FilePipelineStorage(config.outputDir)
        val cache = NoopCache()
        val stepsToSkip = mutableSetOf<String>()

        val state = loadExistingState(outputStorage).toMutableMap()
        if (additionalContext != null) {
            val add = state.getOrPut("additional_context") { mutableMapOf<String, Any?>() }
            if (add is MutableMap<*, *>) {
                @Suppress("UNCHECKED_CAST")
                (add as MutableMap<String, Any?>).putAll(additionalContext)
            }
        }

        val context: PipelineRunContext
        val activeOutput: PipelineStorage
        val previousStorage: PipelineStorage?

        if (isUpdateRun) {
            val updateRoot = FilePipelineStorage(config.updateOutputDir)
            val timestamp = timestamp()
            val timestamped = updateRoot.child(timestamp)
            val deltaStorage = timestamped.child("delta")
            val backup = timestamped.child("previous")
            copyPreviousOutput(outputStorage, backup)
            state["update_timestamp"] = timestamp

            if (inputDocumentsJson != null) {
                deltaStorage.set("documents.json", inputDocumentsJson)
                stepsToSkip.add("load_update_documents")
            }

            context =
                PipelineRunContext(
                    inputStorage = inputStorage,
                    outputStorage = deltaStorage,
                    previousStorage = backup,
                    cache = cache,
                    callbacks = callbacks,
                    state = state,
                )
            activeOutput = deltaStorage
        } else {
            if (inputDocumentsJson != null) {
                outputStorage.set("documents.json", inputDocumentsJson)
                stepsToSkip.add("load_input_documents")
            }

            context =
                PipelineRunContext(
                    inputStorage = inputStorage,
                    outputStorage = outputStorage,
                    cache = cache,
                    callbacks = callbacks,
                    state = state,
                )
            activeOutput = outputStorage
        }

        var lastWorkflow = "<startup>"
        val start = System.nanoTime()
        try {
            dumpJson(context)
            val workflows = pipeline.run().filterNot { it.first in stepsToSkip }.toList()
            val totalWorkflows = workflows.size
            for ((idx, pair) in workflows.withIndex()) {
                val (name, fn) = pair
                context.callbacks.progress(
                    Progress(
                        description = "pipeline ",
                        totalItems = totalWorkflows,
                        completedItems = idx,
                    ),
                )
                lastWorkflow = name
                callbacks.workflowStart(name)
                val wStart = System.nanoTime()
                val result = fn(config, context)
                callbacks.workflowEnd(name, result)
                emit(
                    PipelineRunResult(
                        workflow = name,
                        result = result.result,
                        state = context.state,
                        errors = null,
                    ),
                )
                context.stats.workflows[name] = mapOf("overall" to secondsSince(wStart))
                if (result.stop) {
                    break
                }
            }
            callbacks.progress(
                Progress(
                    description = "pipeline ",
                    totalItems = totalWorkflows,
                    completedItems = totalWorkflows,
                ),
            )
            context.stats.totalRuntime = secondsSince(start)
            dumpJson(context)
        } catch (t: Throwable) {
            emit(
                PipelineRunResult(
                    workflow = lastWorkflow,
                    result = null,
                    state = context.state,
                    errors = listOf(t.message ?: t.javaClass.simpleName),
                ),
            )
        } finally {
            persistStatsAndState(context)
        }
    }

private suspend fun loadExistingState(storage: PipelineStorage): Map<String, Any?> {
    val stateJson = storage.get("context.json") ?: return emptyMap()
    val elementMap: Map<String, kotlinx.serialization.json.JsonElement> = Json.decodeFromString(StateCodec.stateSerializer, stateJson)
    return StateCodec.decodeState(elementMap).toMutableMap()
}

private suspend fun dumpJson(context: PipelineRunContext) {
    val statsSnapshot = StatsSnapshot(context.stats.workflows, context.stats.totalRuntime)
    val statsJson = Json { prettyPrint = true }.encodeToString(statsSnapshot)
    context.outputStorage.set("stats.json", statsJson)

    val temp = context.state.remove("additional_context")
    try {
        val stateJson = Json { prettyPrint = true }.encodeToString(StateCodec.stateSerializer, StateCodec.encodeState(context.state))
        context.outputStorage.set("context.json", stateJson)
    } finally {
        if (temp != null) {
            context.state["additional_context"] = temp
        }
    }
}

private suspend fun persistStatsAndState(context: PipelineRunContext) {
    dumpJson(context)
    context.outputStorage.set("last_run.txt", "completed at ${Instant.now()}")
}

private fun secondsSince(startNano: Long): Double = (System.nanoTime() - startNano) / 1e9

private fun timestamp(): String =
    java.time.format.DateTimeFormatter
        .ofPattern("yyyyMMdd-HHmmss")
        .format(java.time.LocalDateTime.now())

private suspend fun copyPreviousOutput(
    storage: PipelineStorage,
    copyStorage: PipelineStorage,
) {
    for ((relative, _) in storage.find(Regex("\\.parquet$|\\.json$"))) {
        val content = storage.get(relative)
        if (content != null) {
            copyStorage.set(relative, content)
        }
    }
}

@kotlinx.serialization.Serializable
private data class StatsSnapshot(
    val workflows: Map<String, Map<String, Double>>,
    val totalRuntime: Double,
)
