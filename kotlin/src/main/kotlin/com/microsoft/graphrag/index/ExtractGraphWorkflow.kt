package com.microsoft.graphrag.index

import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonIgnoreProperties
import com.fasterxml.jackson.annotation.JsonProperty
import com.microsoft.graphrag.logger.ProgressHandler
import com.microsoft.graphrag.logger.progressTicker
import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.service.AiServices
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.security.MessageDigest
import java.util.UUID

class ExtractGraphWorkflow(
    private val chatModel: OpenAiChatModel,
    private val prompts: PromptRepository = PromptRepository(),
) {
    private val extractor =
        AiServices.create(Extractor::class.java, chatModel)
    private val json = Json { ignoreUnknownKeys = true }

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
    suspend fun extract(
        chunks: List<DocumentChunk>,
        cache: PipelineCache? = null,
        options: ExtractGraphOptions = ExtractGraphOptions(),
        progress: ProgressHandler? = null,
    ): GraphExtractResult {
        val ticker = progressTicker(progress, chunks.size, "extract graph progress: ")

        val results =
            coroutineScope {
                val semaphore = Semaphore(options.maxConcurrency.coerceAtLeast(1))
                val tasks =
                    chunks.map { chunk ->
                        async {
                            semaphore.withPermit {
                                val response = runExtraction(chunk, cache, options)
                                logger.debug { "LLM structured response for chunk ${chunk.id}: $response" }
                                val parsed = parseResponse(response)
                                val entities = parsed.entities.map { it.copy(sourceChunkId = chunk.id) }
                                val relationships = parsed.relationships.map { it.copy(sourceChunkId = chunk.id) }
                                ticker(1)
                                RawGraphExtractResult(entities, relationships)
                            }
                        }
                    }
                tasks.awaitAll()
            }

        val rawEntities = results.flatMap { it.entities }
        val rawRelationships = results.flatMap { it.relationships }

        val mergedEntities = mergeEntities(rawEntities)
        val mergedRelationships = mergeRelationships(rawRelationships, mergedEntities)

        return GraphExtractResult(entities = mergedEntities, relationships = mergedRelationships)
    }

    private suspend fun runExtraction(
        chunk: DocumentChunk,
        cache: PipelineCache?,
        options: ExtractGraphOptions,
    ): ModelExtractionResponse? {
        val prompt = buildPrompt(chunk, options)
        logger.debug { "ExtractGraph: chunk ${chunk.id} text preview: ${chunk.text.take(200)}" }

        if (options.useCache && cache != null) {
            val cached = cache.get(cacheKey(options, prompt))
            if (!cached.isNullOrBlank()) {
                return runCatching { json.decodeFromString<ModelExtractionResponse>(cached) }.getOrNull()
            }
        }

        val response =
            withContext(Dispatchers.IO) {
                invokeChat(prompt)
            }

        if (options.useCache && cache != null && response != null) {
            cache.set(cacheKey(options, prompt), json.encodeToString(response))
        }

        return response
    }

    private fun buildPrompt(
        chunk: DocumentChunk,
        options: ExtractGraphOptions,
    ): String {
        val template = options.promptOverride ?: prompts.loadExtractGraphPrompt()
        val entityTypes =
            options.entityTypes
                .map { it.trim().uppercase() }
                .filter { it.isNotBlank() }
                .ifEmpty { DEFAULT_ENTITY_TYPES }
                .joinToString(",")
        return template
            .replace("{entity_types}", entityTypes)
            .replace("{input_text}", chunk.text)
    }

    private fun cacheKey(
        options: ExtractGraphOptions,
        prompt: String,
    ): String = "${options.cacheKeyPrefix}:${sha256Hex(prompt)}"

    private fun parseResponse(response: ModelExtractionResponse?): RawGraphExtractResult {
        val extraction =
            response ?: return RawGraphExtractResult(emptyList(), emptyList())

        val entities =
            extraction.entities.mapNotNull { entity ->
                val name = entity.name.trim()
                val type = entity.type.trim()
                if (name.isEmpty() || type.isEmpty()) return@mapNotNull null

                RawEntity(
                    name = name,
                    type = type,
                    description = entity.description?.takeIf { it.isNotBlank() }?.trim(),
                    sourceChunkId = "",
                )
            }

        val relationships =
            extraction.relationships.mapNotNull { relationship ->
                val source = relationship.source.trim()
                val target = relationship.target.trim()
                if (source.isEmpty() || target.isEmpty()) return@mapNotNull null

                RawRelationship(
                    sourceName = source,
                    targetName = target,
                    type = relationship.type?.takeIf { it.isNotBlank() } ?: "related_to",
                    description = buildRelationshipDescription(relationship),
                    sourceChunkId = "",
                )
            }

        return RawGraphExtractResult(entities, relationships)
    }

    private fun mergeEntities(rawEntities: List<RawEntity>): List<Entity> {
        if (rawEntities.isEmpty()) return emptyList()

        val grouped =
            rawEntities.groupBy { keyForEntity(it.name, it.type) }

        return grouped.values.map { group ->
            val sample = group.first()
            val descriptions = group.mapNotNull { it.description?.takeIf { d -> d.isNotBlank() } }.distinct()
            val textUnitIds = group.map { it.sourceChunkId }.distinct()
            val frequency = group.size

            Entity(
                id = stableEntityId(sample.name, sample.type),
                name = sample.name,
                type = sample.type,
                sourceChunkId = textUnitIds.firstOrNull().orEmpty(),
                description = descriptions.joinToString(" | ").ifBlank { null },
                textUnitIds = textUnitIds,
                attributes =
                    mapOf(
                        "frequency" to frequency.toString(),
                        "descriptions" to descriptions.joinToString(" | "),
                    ),
            )
        }
    }

    private fun mergeRelationships(
        rawRelationships: List<RawRelationship>,
        mergedEntities: List<Entity>,
    ): List<Relationship> {
        if (rawRelationships.isEmpty()) return emptyList()

        val nameToId = buildEntityNameIndex(mergedEntities)
        val grouped =
            rawRelationships.groupBy { keyForRelationship(it.sourceName, it.targetName) }

        return grouped.values.map { group ->
            val sample = group.first()
            val descriptions = group.mapNotNull { it.description?.takeIf { d -> d.isNotBlank() } }.distinct()
            val textUnitIds = group.map { it.sourceChunkId }.distinct()
            val weight = group.size.toDouble()

            val sourceId = nameToId[sample.sourceName.lowercase()] ?: sample.sourceName
            val targetId = nameToId[sample.targetName.lowercase()] ?: sample.targetName

            Relationship(
                sourceId = sourceId,
                targetId = targetId,
                type = sample.type.ifBlank { "related_to" },
                description = descriptions.joinToString(" | ").ifBlank { null },
                sourceChunkId = textUnitIds.firstOrNull().orEmpty(),
                weight = weight,
                textUnitIds = textUnitIds,
            )
        }
    }

    private fun buildEntityNameIndex(entities: List<Entity>): Map<String, String> {
        val grouped = entities.groupBy { it.name.lowercase() }
        return grouped
            .mapNotNull { (name, list) ->
                if (list.size == 1) name to list.first().id else null
            }.toMap()
    }

    private fun keyForEntity(
        name: String,
        type: String,
    ): String = "${type.trim().lowercase()}|${name.trim().lowercase()}"

    private fun keyForRelationship(
        source: String,
        target: String,
    ): String = "${source.trim().lowercase()}|${target.trim().lowercase()}"

    private fun stableEntityId(
        name: String,
        type: String,
    ): String = UUID.nameUUIDFromBytes("${type.trim().lowercase()}|${name.trim().lowercase()}".toByteArray()).toString()

    private fun sha256Hex(value: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val bytes = digest.digest(value.toByteArray())
        val sb = StringBuilder(bytes.size * 2)
        for (b in bytes) {
            sb.append(String.format("%02x", b))
        }
        return sb.toString()
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
        val DEFAULT_ENTITY_TYPES = listOf("ORGANIZATION", "PERSON", "GPE", "LOCATION")
    }
}

data class ExtractGraphOptions(
    val entityTypes: List<String> = ExtractGraphWorkflow.DEFAULT_ENTITY_TYPES,
    val maxConcurrency: Int = 4,
    val useCache: Boolean = true,
    val cacheKeyPrefix: String = "extract_graph",
    val promptOverride: String? = null,
)

private data class RawGraphExtractResult(
    val entities: List<RawEntity>,
    val relationships: List<RawRelationship>,
)

private data class RawEntity(
    val name: String,
    val type: String,
    val description: String?,
    val sourceChunkId: String,
)

private data class RawRelationship(
    val sourceName: String,
    val targetName: String,
    val type: String,
    val description: String?,
    val sourceChunkId: String,
)

@Serializable
@JsonIgnoreProperties(ignoreUnknown = true)
data class ModelExtractionResponse
    @JsonCreator
    constructor(
        @param:JsonProperty("entities") val entities: List<ModelEntity> = emptyList(),
        @param:JsonProperty("relationships") val relationships: List<ModelRelationship> = emptyList(),
    )

@Serializable
@JsonIgnoreProperties(ignoreUnknown = true)
data class ModelEntity
    @JsonCreator
    constructor(
        @param:JsonProperty("id") val id: String? = null,
        @param:JsonProperty("name") val name: String,
        @param:JsonProperty("type") val type: String,
        @param:JsonProperty("description") val description: String? = null,
    )

@Serializable
@JsonIgnoreProperties(ignoreUnknown = true)
data class ModelRelationship
    @JsonCreator
    constructor(
        @param:JsonProperty("source") val source: String,
        @param:JsonProperty("target") val target: String,
        @param:JsonProperty("type") val type: String? = null,
        @param:JsonProperty("description") val description: String? = null,
        @param:JsonProperty("strength") val strength: Double? = null,
    )
