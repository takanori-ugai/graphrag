package com.microsoft.graphrag.index

import com.fasterxml.jackson.annotation.JsonIgnoreProperties
import com.fasterxml.jackson.annotation.JsonProperty
import com.fasterxml.jackson.databind.ObjectMapper
import dev.langchain4j.data.message.ChatMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.openai.OpenAiChatModel
import io.github.oshai.kotlinlogging.KotlinLogging

class ExtractGraphWorkflow(
    private val chatModel: OpenAiChatModel,
    private val prompts: PromptRepository = PromptRepository(),
    private val objectMapper: ObjectMapper = ObjectMapper().findAndRegisterModules(),
) {
    /**
     * Extracts entities and relationships from the provided document chunks using the configured extraction model.
     *
     * The prompt requests JSON output with entities and relationships; the response is parsed into
     * aggregated entities and relationships.
     */
    suspend fun extract(
        chunks: List<DocumentChunk>,
        entityTypes: List<String> = DEFAULT_ENTITY_TYPES,
    ): GraphExtractResult {
        val entities = linkedMapOf<String, EntityAccumulator>()
        val relationships = linkedMapOf<Pair<String, String>, RelationshipAccumulator>()

        val entityTypesValue = entityTypes.joinToString(",")
        val promptTemplate = prompts.loadExtractGraphPrompt()

        for (chunk in chunks) {
            val prompt =
                promptTemplate
                    .replace("{entity_types}", entityTypesValue)
                    .replace("{input_text}", chunk.text)
            logger.debug { "ExtractGraph: chunk ${chunk.id} text preview: ${chunk.text.take(200)}" }
            val parsed =
                parseJson(
                    extractJson(prompt),
                    sourceChunkId = chunk.id,
                    entities = entities,
                    relationships = relationships,
                )
            logger.debug { "ExtractGraph parsed for chunk ${chunk.id}: entities=${parsed.first}, relationships=${parsed.second}" }
        }

        return GraphExtractResult(
            entities = entities.values.map { it.toEntity() },
            relationships = relationships.values.map { it.toRelationship() },
        )
    }

    private fun extractJson(prompt: String): String {
        val messages =
            listOf<ChatMessage>(
                UserMessage(prompt),
            )
        return chat(messages)
    }

    private fun chat(messages: List<ChatMessage>): String =
        runCatching { chatModel.chat(messages).aiMessage().text() }
            .getOrElse {
                logger.warn { "Graph extraction chat failed: ${it.message}" }
                ""
            }

    private fun parseJson(
        raw: String,
        sourceChunkId: String,
        entities: MutableMap<String, EntityAccumulator>,
        relationships: MutableMap<Pair<String, String>, RelationshipAccumulator>,
    ): Pair<Int, Int> {
        if (raw.isBlank()) return 0 to 0
        val json = extractJsonObject(raw) ?: return 0 to 0
        val parsed =
            runCatching { objectMapper.readValue(json, GraphExtractionResponse::class.java) }
                .getOrElse {
                    logger.warn { "Graph extraction JSON decode failed: ${it.message}" }
                    return 0 to 0
                }
        var entityCount = 0
        var relationshipCount = 0
        parsed.entities.forEach { entity ->
            val name = cleanString(entity.name).uppercase()
            val type = cleanString(entity.type).uppercase()
            val description = cleanString(entity.description ?: "")
            if (name.isBlank()) return@forEach
            val acc = entities.getOrPut(name) { EntityAccumulator(name = name) }
            acc.type = if (type.isNotBlank()) type else acc.type
            acc.addDescription(description)
            acc.addSourceChunkId(sourceChunkId)
            entityCount++
        }

        parsed.relationships.forEach { rel ->
            val source = cleanString(rel.source).uppercase()
            val target = cleanString(rel.target).uppercase()
            val type = cleanString(rel.type ?: "related_to").uppercase()
            val description = cleanString(rel.description ?: "")
            val weight = rel.strength ?: 1.0
            if (source.isBlank() || target.isBlank()) return@forEach
            val key = source to target
            val acc = relationships.getOrPut(key) { RelationshipAccumulator(source = source, target = target) }
            acc.addType(type)
            acc.addDescription(description)
            acc.addSourceChunkId(sourceChunkId)
            acc.weight += weight
            relationshipCount++
        }

        return entityCount to relationshipCount
    }

    private fun cleanString(value: String): String =
        value.trim().replace(CONTROL_CHAR_REGEX, "")

    private fun extractJsonObject(text: String): String? {
        val fenced = extractFromCodeFence(text)
        if (fenced != null) return fenced
        return extractBalancedJson(text)
    }

    private fun extractFromCodeFence(text: String): String? {
        val fenceRegex = Regex("```(?:json)?\\s*([\\s\\S]*?)\\s*```", RegexOption.IGNORE_CASE)
        val match =
            fenceRegex
                .findAll(text)
                .firstOrNull { it.groupValues[1].contains('{') }
                ?: return null
        return extractBalancedJson(match.groupValues[1])
    }

    private fun extractBalancedJson(text: String): String? {
        val start = text.indexOf('{')
        if (start == -1) return null
        var depth = 0
        var inString = false
        var escape = false
        for (i in start until text.length) {
            val c = text[i]
            if (inString) {
                if (escape) {
                    escape = false
                } else {
                    when (c) {
                        '\\' -> escape = true
                        '"' -> inString = false
                    }
                }
                continue
            }
            when (c) {
                '"' -> inString = true
                '{' -> depth++
                '}' -> {
                    depth--
                    if (depth == 0) return text.substring(start, i + 1)
                }
            }
        }
        return null
    }

    private data class EntityAccumulator(
        val name: String,
        var type: String = "",
    ) {
        private val descriptions = linkedSetOf<String>()
        private val sourceChunkIds = linkedSetOf<String>()

        fun addDescription(description: String) {
            if (description.isNotBlank()) descriptions += description
        }

        fun addSourceChunkId(chunkId: String) {
            if (chunkId.isNotBlank()) sourceChunkIds += chunkId
        }

        fun toEntity(): Entity =
            Entity(
                id = name,
                name = name,
                type = type,
                description = descriptions.joinToString("\n").ifBlank { null },
                sourceChunkId = sourceChunkIds.joinToString(", "),
            )
    }

    private data class RelationshipAccumulator(
        val source: String,
        val target: String,
    ) {
        var type: String = "related_to"
        var weight: Double = 0.0
        private val descriptions = linkedSetOf<String>()
        private val sourceChunkIds = linkedSetOf<String>()

        fun addType(value: String) {
            if (value.isNotBlank()) type = value
        }

        fun addDescription(description: String) {
            if (description.isNotBlank()) descriptions += description
        }

        fun addSourceChunkId(chunkId: String) {
            if (chunkId.isNotBlank()) sourceChunkIds += chunkId
        }

        fun toRelationship(): Relationship =
            Relationship(
                sourceId = source,
                targetId = target,
                type = type,
                description = descriptions.joinToString("\n").ifBlank { null },
                sourceChunkId = sourceChunkIds.joinToString(", "),
                weight = weight,
            )
    }

    companion object {
        private val logger = KotlinLogging.logger {}
        private val CONTROL_CHAR_REGEX = Regex("[\\x00-\\x1f\\x7f-\\x9f]")
        private val DEFAULT_ENTITY_TYPES = listOf("organization", "person", "geo", "event")
    }
}

@JsonIgnoreProperties(ignoreUnknown = true)
private data class GraphExtractionResponse(
    @param:JsonProperty("entities") val entities: List<GraphEntity> = emptyList(),
    @param:JsonProperty("relationships") val relationships: List<GraphRelationship> = emptyList(),
)

@JsonIgnoreProperties(ignoreUnknown = true)
private data class GraphEntity(
    @param:JsonProperty("name") val name: String,
    @param:JsonProperty("type") val type: String,
    @param:JsonProperty("description") val description: String? = null,
)

@JsonIgnoreProperties(ignoreUnknown = true)
private data class GraphRelationship(
    @param:JsonProperty("source") val source: String,
    @param:JsonProperty("target") val target: String,
    @param:JsonProperty("description") val description: String? = null,
    @param:JsonProperty("strength") val strength: Double? = null,
    @param:JsonProperty("type") val type: String? = null,
)
