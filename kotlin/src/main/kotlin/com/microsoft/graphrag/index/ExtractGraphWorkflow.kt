package com.microsoft.graphrag.index

import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonIgnoreProperties
import com.fasterxml.jackson.annotation.JsonProperty
import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.service.AiServices
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.Serializable
import java.util.UUID

class ExtractGraphWorkflow(
    private val chatModel: OpenAiChatModel,
    private val prompts: PromptRepository = PromptRepository(),
) {
    private val extractor =
        AiServices.create(Extractor::class.java, chatModel)

    /**
     * Extracts entities and relationships from the provided document chunks using the configured extraction model.
     *
     * For each chunk this function builds a prompt, invokes the model, parses the structured response,
     * normalizes entity ids
     * (generating UUIDs when absent), assigns each entity and relationship a sourceChunkId equal to the
     * originating chunk's id,
     * and resolves relationship endpoints to entity ids when an entity name in the same chunk matches a
     * relationship endpoint.
     *
     * @param chunks The list of DocumentChunk objects to analyze; each chunk's id is recorded on entities and
     * relationships produced from that chunk.
     * @return A GraphExtractResult containing all aggregated entities and relationships extracted from the
     * input chunks.
     */
    suspend fun extract(chunks: List<DocumentChunk>): GraphExtractResult {
        val entities = mutableListOf<Entity>()
        val relationships = mutableListOf<Relationship>()

        for (chunk in chunks) {
            val prompt = buildPrompt(chunk)
            logger.debug { "ExtractGraph: chunk ${chunk.id} text preview: ${chunk.text.take(200)}" }
            val response = invokeChat(prompt)
            logger.debug { "LLM structured response for chunk ${chunk.id}: $response" }
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

    private fun parseResponse(response: ModelExtractionResponse?): GraphExtractResult {
        val extraction =
            response ?: return GraphExtractResult(emptyList(), emptyList())

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

    /**
     * Sends the given prompt to the extractor and returns the extraction result.
     *
     * @param prompt The user message / prompt sent to the extractor.
     * @return `ModelExtractionResponse` returned by the extractor, or `null` if the extraction failed.
     */
    private fun invokeChat(prompt: String): ModelExtractionResponse? =
        runCatching { extractor.extract(prompt) }.getOrElse {
            logger.warn(it) {
                "Failed to parse extraction response; prompt preview: ${prompt.take(200)}"
            }
            null
        }

    private interface Extractor {
        /**
         * Extracts entities and relationships from the given user message according to the extraction instructions.
         *
         * @param userMessage The user-facing prompt or document text containing extraction instructions and
         * content to analyze.
         * @return A `ModelExtractionResponse` containing the extracted entities and relationships; lists may be
         * empty if none are found.
         */
        @dev.langchain4j.service.SystemMessage(
            "You are an information extraction assistant. Extract entities and relationships exactly as instructed in the user message.",
        )
        fun extract(
            @dev.langchain4j.service.UserMessage userMessage: String,
        ): ModelExtractionResponse
    }

    companion object {
        private val logger = KotlinLogging.logger {}
    }
}

@Serializable
@JsonIgnoreProperties(ignoreUnknown = true)
data class ModelExtractionResponse
    @JsonCreator
    constructor(
        @JsonProperty("entities") val entities: List<ModelEntity> = emptyList(),
        @JsonProperty("relationships") val relationships: List<ModelRelationship> = emptyList(),
    )

@Serializable
@JsonIgnoreProperties(ignoreUnknown = true)
data class ModelEntity
    @JsonCreator
    constructor(
        @JsonProperty("id") val id: String? = null,
        @JsonProperty("name") val name: String,
        @JsonProperty("type") val type: String,
        @JsonProperty("description") val description: String? = null,
    )

@Serializable
@JsonIgnoreProperties(ignoreUnknown = true)
data class ModelRelationship
    @JsonCreator
    constructor(
        @JsonProperty("source") val source: String,
        @JsonProperty("target") val target: String,
        @JsonProperty("type") val type: String? = null,
        @JsonProperty("description") val description: String? = null,
        @JsonProperty("strength") val strength: Double? = null,
    )
