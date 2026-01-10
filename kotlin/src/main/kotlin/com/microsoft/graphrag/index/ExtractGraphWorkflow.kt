package com.microsoft.graphrag.index

import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.openai.OpenAiChatModel
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import java.util.UUID

class ExtractGraphWorkflow(
    private val chatModel: OpenAiChatModel,
) {
    private val json = Json { ignoreUnknownKeys = true }

    suspend fun extract(chunks: List<DocumentChunk>): GraphExtractResult {
        val entities = mutableListOf<Entity>()
        val relationships = mutableListOf<Relationship>()

        for (chunk in chunks) {
            val prompt = buildPrompt(chunk)
            val response = invokeChat(prompt)
            val parsed = parseResponse(response)
            val chunkEntities =
                parsed.entities.map {
                    it.copy(
                        id = it.id.ifBlank { UUID.randomUUID().toString() },
                        sourceChunkId = chunk.id,
                    )
                }
            val chunkRelationships =
                parsed.relationships.map {
                    it.copy(
                        sourceChunkId = chunk.id,
                    )
                }
            entities += chunkEntities
            relationships += chunkRelationships
        }

        return GraphExtractResult(entities = entities, relationships = relationships)
    }

    private fun buildPrompt(chunk: DocumentChunk): String =
        """
        Extract entities and relationships from the following text. 
        Return JSON with fields "entities" and "relationships".
        Entities: [{ "id": string, "name": string, "type": string }]
        Relationships: [{ "sourceId": string, "targetId": string, "type": string, "description": string }]
        Text:
        ${chunk.text}
        """.trimIndent()

    private fun parseResponse(content: String): GraphExtractResult =
        try {
            json.decodeFromString(ModelExtractionResponse.serializer(), content).toGraphResult()
        } catch (_: Exception) {
            GraphExtractResult(emptyList(), emptyList())
        }

    private fun invokeChat(prompt: String): String {
        val message = UserMessage.from(prompt)
        val method =
            chatModel.javaClass.methods.firstOrNull { it.name == "generate" && it.parameterTypes.size == 1 }
        val result = method?.invoke(chatModel, listOf(message))
        return when (result) {
            is dev.langchain4j.data.message.AiMessage -> result.text()
            is dev.langchain4j.model.output.Response<*> -> result.content().toString()
            else -> result?.toString() ?: ""
        }
    }
}

@Serializable
private data class ModelExtractionResponse(
    val entities: List<SerializableEntity> = emptyList(),
    val relationships: List<SerializableRelationship> = emptyList(),
) {
    fun toGraphResult(): GraphExtractResult =
        GraphExtractResult(
            entities =
                entities.map {
                    Entity(
                        id = it.id.ifBlank { UUID.randomUUID().toString() },
                        name = it.name,
                        type = it.type,
                        sourceChunkId = it.sourceChunkId.ifBlank { "" },
                    )
                },
            relationships =
                relationships.map {
                    Relationship(
                        sourceId = it.sourceId,
                        targetId = it.targetId,
                        type = it.type,
                        description = it.description,
                        sourceChunkId = it.sourceChunkId.ifBlank { "" },
                    )
                },
        )
}

@Serializable
private data class SerializableEntity(
    val id: String = "",
    val name: String = "",
    val type: String = "",
    val sourceChunkId: String = "",
)

@Serializable
private data class SerializableRelationship(
    val sourceId: String = "",
    val targetId: String = "",
    val type: String = "",
    val description: String? = null,
    val sourceChunkId: String = "",
)
