package com.microsoft.graphrag.index

import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.service.AiServices
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import java.util.UUID

class ExtractGraphWorkflow(
    private val chatModel: OpenAiChatModel,
    private val prompts: PromptRepository = PromptRepository(),
) {
    private val extractor =
        AiServices.create(Extractor::class.java, chatModel)
    private val json = Json { ignoreUnknownKeys = true }

    suspend fun extract(chunks: List<DocumentChunk>): GraphExtractResult {
        val entities = mutableListOf<Entity>()
        val relationships = mutableListOf<Relationship>()

        for (chunk in chunks) {
            val prompt = buildPrompt(chunk)
            println("ExtractGraph: chunk ${chunk.id} text preview: ${chunk.text.take(200)}")
            val response = invokeChat(prompt)
            println("LLM raw response for chunk ${chunk.id}:\n$response")
            val parsed = parseResponse(response)
            val chunkEntities =
                parsed.entities.map { entity ->
                    val name = entity.name.trim()
                    val id = name.ifBlank { entity.id ?: UUID.randomUUID().toString() }
                    entity.copy(
                        id = id,
                        sourceChunkId = chunk.id,
                    )
                }
            val nameToId = chunkEntities.associate { it.name.trim() to it.id }
            val chunkRelationships =
                parsed.relationships.map { rel ->
                    val source = nameToId[rel.sourceId.trim()] ?: rel.sourceId.trim()
                    val target = nameToId[rel.targetId.trim()] ?: rel.targetId.trim()
                    rel.copy(
                        sourceId = source,
                        targetId = target,
                        sourceChunkId = chunk.id,
                    )
                }
            entities += chunkEntities
            relationships += chunkRelationships
        }

        return GraphExtractResult(entities = entities, relationships = relationships)
    }

    private fun buildPrompt(chunk: DocumentChunk): String =
        prompts
            .loadExtractGraphPrompt()
            .replace("{entity_types}", "ORGANIZATION,PERSON,GPE,LOCATION")
            .replace("{input_text}", chunk.text)

    private fun parseResponse(content: String): GraphExtractResult {
        val jsonContent =
            content
                .trim()
                .removePrefix("```json")
                .removePrefix("```")
                .removeSuffix("```")
                .trim()

        val extraction =
            runCatching {
                json.decodeFromString<ModelExtractionResponse>(jsonContent)
            }.getOrElse {
                println("Failed to parse extraction response: ${it.message}")
                return GraphExtractResult(emptyList(), emptyList())
            }

        val entities =
            extraction.entities.mapNotNull { entity ->
                val name = entity.name.trim()
                val type = entity.type.trim()
                if (name.isEmpty() || type.isEmpty()) return@mapNotNull null

                Entity(
                    id = entity.id?.takeIf { it.isNotBlank() } ?: UUID.randomUUID().toString(),
                    name = name,
                    type = type,
                    sourceChunkId = "",
                )
            }

        val relationships =
            extraction.relationships.mapNotNull { relationship ->
                val source = relationship.source.trim()
                val target = relationship.target.trim()
                if (source.isEmpty() || target.isEmpty()) return@mapNotNull null

                Relationship(
                    sourceId = source,
                    targetId = target,
                    type = relationship.type?.takeIf { it.isNotBlank() } ?: "related_to",
                    description = buildRelationshipDescription(relationship),
                    sourceChunkId = "",
                )
            }

        return GraphExtractResult(entities, relationships)
    }

    private fun buildRelationshipDescription(relationship: ModelRelationship): String? {
        val parts = mutableListOf<String>()
        relationship.strength?.let { parts += "strength=$it" }
        relationship.description?.takeIf { it.isNotBlank() }?.let { parts += it.trim() }
        return parts.joinToString("; ").ifBlank { null }
    }

    private fun invokeChat(prompt: String): String = extractor.chat(prompt)

    private interface Extractor {
        @dev.langchain4j.service.SystemMessage(
            "You are an information extraction assistant. Extract entities and relationships exactly as instructed in the user message.",
        )
        fun chat(
            @dev.langchain4j.service.UserMessage userMessage: String,
        ): String
    }
}

@Serializable
private data class ModelExtractionResponse(
    val entities: List<ModelEntity> = emptyList(),
    val relationships: List<ModelRelationship> = emptyList(),
)

@Serializable
private data class ModelEntity(
    val id: String? = null,
    val name: String,
    val type: String,
    val description: String? = null,
)

@Serializable
private data class ModelRelationship(
    val source: String,
    val target: String,
    val type: String? = null,
    val description: String? = null,
    val strength: Double? = null,
)
