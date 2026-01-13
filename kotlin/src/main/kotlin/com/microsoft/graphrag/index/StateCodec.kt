package com.microsoft.graphrag.index

import kotlinx.serialization.KSerializer
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.MapSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/**
 * Utility to serialize and deserialize pipeline state snapshots stored in context.json.
 */
object StateCodec {
    val stateSerializer: KSerializer<Map<String, JsonElement>> =
        MapSerializer(String.serializer(), JsonElement.serializer())

    fun encodeState(state: Map<String, Any?>): Map<String, JsonElement> {
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
                                json.encodeToJsonElement(
                                    ListSerializer(DocumentChunk.serializer()),
                                    value.filterIsInstance<DocumentChunk>(),
                                )
                        }

                        value.all { it is TextUnit } -> {
                            result[key] =
                                json.encodeToJsonElement(ListSerializer(TextUnit.serializer()), value.filterIsInstance<TextUnit>())
                        }

                        value.all { it is Entity } -> {
                            result[key] =
                                json.encodeToJsonElement(ListSerializer(Entity.serializer()), value.filterIsInstance<Entity>())
                        }

                        value.all { it is Relationship } -> {
                            result[key] =
                                json.encodeToJsonElement(ListSerializer(Relationship.serializer()), value.filterIsInstance<Relationship>())
                        }

                        value.all { it is Claim } -> {
                            result[key] =
                                json.encodeToJsonElement(ListSerializer(Claim.serializer()), value.filterIsInstance<Claim>())
                        }

                        value.all { it is TextEmbedding } -> {
                            result[key] =
                                json.encodeToJsonElement(
                                    ListSerializer(TextEmbedding.serializer()),
                                    value.filterIsInstance<TextEmbedding>(),
                                )
                        }

                        value.all { it is EntityEmbedding } -> {
                            result[key] =
                                json.encodeToJsonElement(
                                    ListSerializer(EntityEmbedding.serializer()),
                                    value.filterIsInstance<EntityEmbedding>(),
                                )
                        }

                        value.all { it is CommunityReportEmbedding } -> {
                            result[key] =
                                json.encodeToJsonElement(
                                    ListSerializer(CommunityReportEmbedding.serializer()),
                                    value.filterIsInstance<CommunityReportEmbedding>(),
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
                                json.encodeToJsonElement(
                                    ListSerializer(EntitySummary.serializer()),
                                    value.filterIsInstance<EntitySummary>(),
                                )
                        }
                    }
                }

                is Map<*, *> -> {
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

    fun decodeState(encoded: Map<String, JsonElement>): Map<String, Any?> {
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

                "community_report_embeddings" -> {
                    if (value is JsonArray) {
                        out[key] =
                            json.decodeFromJsonElement(
                                ListSerializer(CommunityReportEmbedding.serializer()),
                                value,
                            )
                    }
                }

                "covariates" -> {
                    if (value is JsonObject) {
                        val covariateMap = mutableMapOf<String, List<Covariate>>()
                        value.forEach { (covariateType, covariates) ->
                            if (covariates is JsonArray) {
                                covariateMap[covariateType] =
                                    json.decodeFromJsonElement(ListSerializer(Covariate.serializer()), covariates)
                            }
                        }
                        out[key] = covariateMap
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
                        val map =
                            value.mapValues { entry ->
                                (entry.value as? JsonPrimitive)?.content ?: entry.value.toString()
                            }
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
}
