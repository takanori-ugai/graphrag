package com.microsoft.graphrag.index

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.MapSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
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
            for ((name, fn) in pipeline.run().filterNot { it.first in stepsToSkip }) {
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
    val elementMap = Json.decodeFromString(stateSerializer, stateJson)
    return decodeState(elementMap).toMutableMap()
}

private suspend fun dumpJson(context: PipelineRunContext) {
    val statsSnapshot = StatsSnapshot(context.stats.workflows, context.stats.totalRuntime)
    val statsJson = Json { prettyPrint = true }.encodeToString(statsSnapshot)
    context.outputStorage.set("stats.json", statsJson)

    val temp = context.state.remove("additional_context")
    try {
        val stateJson = Json { prettyPrint = true }.encodeToString(stateSerializer, encodeState(context.state))
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

/**
 * Basic serializer for Map<String, Any?> to avoid depending on a full JSON tree library.
 */
private val stateSerializer = MapSerializer(String.serializer(), JsonElement.serializer())

private fun encodeState(state: Map<String, Any?>): Map<String, JsonElement> {
    val json = Json { prettyPrint = false }
    val result = mutableMapOf<String, JsonElement>()
    state.forEach { (key, value) ->
        when (value) {
            null -> {
                result[key] = JsonNull
            }

            is List<*> -> {
                when {
                    value.all { it is DocumentChunk } -> {
                        result[key] =
                            json.encodeToJsonElement(ListSerializer(DocumentChunk.serializer()), value.filterIsInstance<DocumentChunk>())
                    }

                    value.all { it is TextUnit } -> {
                        result[key] = json.encodeToJsonElement(ListSerializer(TextUnit.serializer()), value.filterIsInstance<TextUnit>())
                    }

                    value.all { it is Entity } -> {
                        result[key] = json.encodeToJsonElement(ListSerializer(Entity.serializer()), value.filterIsInstance<Entity>())
                    }

                    value.all { it is Relationship } -> {
                        result[key] =
                            json.encodeToJsonElement(ListSerializer(Relationship.serializer()), value.filterIsInstance<Relationship>())
                    }

                    value.all { it is Claim } -> {
                        result[key] = json.encodeToJsonElement(ListSerializer(Claim.serializer()), value.filterIsInstance<Claim>())
                    }

                    value.all { it is TextEmbedding } -> {
                        result[key] =
                            json.encodeToJsonElement(ListSerializer(TextEmbedding.serializer()), value.filterIsInstance<TextEmbedding>())
                    }

                    value.all { it is EntityEmbedding } -> {
                        result[key] =
                            json.encodeToJsonElement(
                                ListSerializer(EntityEmbedding.serializer()),
                                value.filterIsInstance<EntityEmbedding>(),
                            )
                    }

                    value.all { it is CommunityAssignment } -> {
                        result[key] =
                            json.encodeToJsonElement(
                                ListSerializer(CommunityAssignment.serializer()),
                                value.filterIsInstance<CommunityAssignment>(),
                            )
                    }

                    value.all { it is CommunityReport } -> {
                        result[key] =
                            json.encodeToJsonElement(
                                ListSerializer(CommunityReport.serializer()),
                                value.filterIsInstance<CommunityReport>(),
                            )
                    }

                    value.all { it is EntitySummary } -> {
                        result[key] =
                            json.encodeToJsonElement(ListSerializer(EntitySummary.serializer()), value.filterIsInstance<EntitySummary>())
                    }
                }
            }

            is Map<*, *> -> {
                // Only serialize primitive map structures (e.g., community hierarchy)
                if (value.keys.all { it is Int } && value.values.all { it is Int }) {
                    @Suppress("UNCHECKED_CAST")
                    result[key] = json.encodeToJsonElement(MapSerializer(Int.serializer(), Int.serializer()), value as Map<Int, Int>)
                } else if (value.keys.all { it is String } && value.values.all { it is String }) {
                    @Suppress("UNCHECKED_CAST")
                    result[key] =
                        json.encodeToJsonElement(MapSerializer(String.serializer(), String.serializer()), value as Map<String, String>)
                }
            }

            is String -> {
                result[key] = JsonPrimitive(value)
            }

            is Number -> {
                result[key] = JsonPrimitive(value)
            }

            is Boolean -> {
                result[key] = JsonPrimitive(value)
            }

            else -> {
                // skip non-serializable state entries (e.g., graph instances)
            }
        }
    }
    return result
}

private fun decodeState(encoded: Map<String, JsonElement>): Map<String, Any?> {
    val json = Json { ignoreUnknownKeys = true }
    val out = mutableMapOf<String, Any?>()
    encoded.forEach { (key, value) ->
        when (key) {
            "chunks" -> {
                if (value is JsonArray) {
                    out[key] = json.decodeFromJsonElement(ListSerializer(DocumentChunk.serializer()), value)
                }
            }

            "text_units" -> {
                if (value is JsonArray) {
                    out[key] = json.decodeFromJsonElement(ListSerializer(TextUnit.serializer()), value)
                }
            }

            "entities" -> {
                if (value is JsonArray) {
                    out[key] = json.decodeFromJsonElement(ListSerializer(Entity.serializer()), value)
                }
            }

            "relationships" -> {
                if (value is JsonArray) {
                    out[key] = json.decodeFromJsonElement(ListSerializer(Relationship.serializer()), value)
                }
            }

            "claims" -> {
                if (value is JsonArray) {
                    out[key] = json.decodeFromJsonElement(ListSerializer(Claim.serializer()), value)
                }
            }

            "text_embeddings" -> {
                if (value is JsonArray) {
                    out[key] = json.decodeFromJsonElement(ListSerializer(TextEmbedding.serializer()), value)
                }
            }

            "entity_embeddings" -> {
                if (value is JsonArray) {
                    out[key] = json.decodeFromJsonElement(ListSerializer(EntityEmbedding.serializer()), value)
                }
            }

            "communities" -> {
                if (value is JsonArray) {
                    out[key] = json.decodeFromJsonElement(ListSerializer(CommunityAssignment.serializer()), value)
                }
            }

            "community_reports" -> {
                if (value is JsonArray) {
                    out[key] = json.decodeFromJsonElement(ListSerializer(CommunityReport.serializer()), value)
                }
            }

            "entity_summaries" -> {
                if (value is JsonArray) {
                    out[key] = json.decodeFromJsonElement(ListSerializer(EntitySummary.serializer()), value)
                }
            }

            "community_hierarchy" -> {
                if (value is JsonObject) {
                    out[key] = json.decodeFromJsonElement(MapSerializer(Int.serializer(), Int.serializer()), value)
                }
            }

            "additional_context" -> {
                if (value is JsonObject) {
                    val map = value.mapValues { it.value.toString() }
                    out[key] = map
                }
            }

            else -> {
                // leave unsupported keys out; non-serializable entries (like graphs) are not persisted
            }
        }
    }
    return out
}

@kotlinx.serialization.Serializable
private data class StatsSnapshot(
    val workflows: Map<String, Map<String, Double>>,
    val totalRuntime: Double,
)
