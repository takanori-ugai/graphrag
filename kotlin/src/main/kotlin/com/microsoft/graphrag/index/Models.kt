package com.microsoft.graphrag.index

import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonIgnoreProperties
import com.fasterxml.jackson.annotation.JsonProperty
import kotlinx.serialization.Serializable

/**
 * Covariate record associated with an entity or claim.
 *
 * @property id Covariate identifier.
 * @property subjectId Identifier of the subject this covariate belongs to.
 * @property covariateType Type label for the covariate.
 * @property attributes Additional covariate attributes.
 */
@Serializable
data class Covariate(
    val id: String,
    val subjectId: String,
    val covariateType: String = "covariate",
    val attributes: Map<String, String> = emptyMap(),
)

/**
 * Raw document chunk extracted from source content.
 *
 * @property id Chunk identifier.
 * @property sourcePath Path to the source document.
 * @property text Chunk text content.
 */
@Serializable
data class DocumentChunk(
    val id: String,
    val sourcePath: String,
    val text: String,
)

/**
 * Extracted entity record.
 *
 * @property id Entity identifier.
 * @property name Entity name.
 * @property type Entity type label.
 * @property sourceChunkId Chunk id where the entity was found.
 * @property description Optional description text.
 * @property rank Optional ranking score.
 * @property shortId Optional short identifier.
 * @property communityIds Community ids associated with the entity.
 * @property textUnitIds Text unit ids associated with the entity.
 * @property attributes Additional entity attributes.
 */
@Serializable
data class Entity(
    val id: String,
    val name: String,
    val type: String,
    val sourceChunkId: String,
    val description: String? = null,
    val rank: Double? = 1.0,
    val shortId: String? = null,
    val communityIds: List<Int> = emptyList(),
    val textUnitIds: List<String> = emptyList(),
    val attributes: Map<String, String> = emptyMap(),
)

/**
 * Relationship between two entities.
 *
 * @property sourceId Source entity id.
 * @property targetId Target entity id.
 * @property type Relationship type label.
 * @property description Optional relationship description.
 * @property sourceChunkId Chunk id where the relationship was found.
 * @property weight Optional relationship weight.
 * @property rank Optional relationship rank.
 * @property attributes Additional relationship attributes.
 * @property shortId Optional short identifier.
 * @property id Optional relationship identifier.
 * @property textUnitIds Text unit ids associated with the relationship.
 */
@Serializable
data class Relationship(
    val sourceId: String,
    val targetId: String,
    val type: String,
    val description: String? = null,
    val sourceChunkId: String,
    val weight: Double? = 1.0,
    val rank: Double? = 1.0,
    val attributes: Map<String, String> = emptyMap(),
    val shortId: String? = null,
    val id: String? = null,
    val textUnitIds: List<String> = emptyList(),
)

/**
 * Embedding for a text chunk.
 *
 * @property chunkId Chunk identifier.
 * @property vector Embedding vector.
 */
@Serializable
data class TextEmbedding(
    val chunkId: String,
    val vector: List<Double>,
)

/**
 * Embedding for an entity.
 *
 * @property entityId Entity identifier.
 * @property vector Embedding vector.
 */
@Serializable
data class EntityEmbedding(
    val entityId: String,
    val vector: List<Double>,
)

/**
 * Embedding for a community report.
 *
 * @property communityId Community identifier.
 * @property vector Embedding vector.
 */
@Serializable
data class CommunityReportEmbedding(
    val communityId: Int,
    val vector: List<Double>,
)

/**
 * Assignment mapping an entity to a community.
 *
 * @property entityId Entity identifier.
 * @property communityId Community identifier.
 */
@Serializable
data class CommunityAssignment(
    val entityId: String,
    val communityId: Int,
)

/**
 * Result of community detection.
 *
 * @property assignments Entity-to-community assignments.
 * @property hierarchy Community hierarchy mapping.
 */
@Serializable
data class CommunityDetectionResult(
    val assignments: List<CommunityAssignment>,
    val hierarchy: Map<Int, Int>,
)

/**
 * Summary report for a community.
 *
 * @property communityId Community identifier.
 * @property summary Summary text.
 * @property parentCommunityId Optional parent community id.
 * @property id Optional report identifier.
 * @property title Optional report title.
 * @property rank Optional community rank.
 * @property fullContent Optional full report content.
 * @property shortId Optional short identifier.
 * @property attributes Additional report attributes.
 */
@Serializable
data class CommunityReport(
    val communityId: Int,
    val summary: String,
    val parentCommunityId: Int? = null,
    val id: String? = null,
    val title: String? = null,
    val rank: Double? = 1.0,
    val fullContent: String? = null,
    val shortId: String? = null,
    val attributes: Map<String, String> = emptyMap(),
)

/**
 * Claim extracted from source material.
 *
 * Uses both kotlinx serialization (internal encoding) and Jackson annotations (LangChain4j responses).
 *
 * @property subject Claim subject.
 * @property object Claim object.
 * @property claimType Claim type label.
 * @property status Claim status.
 * @property startDate Claim start date.
 * @property endDate Claim end date.
 * @property description Claim description.
 * @property sourceText Source text backing the claim.
 */
@Serializable
@JsonIgnoreProperties(ignoreUnknown = true)
data class Claim
    @JsonCreator
    constructor(
        @param:JsonProperty("subject") val subject: String,
        @param:JsonProperty("object") val `object`: String,
        @param:JsonProperty("claimType") val claimType: String,
        @param:JsonProperty("status") val status: String,
        @param:JsonProperty("startDate") val startDate: String,
        @param:JsonProperty("endDate") val endDate: String,
        @param:JsonProperty("description") val description: String,
        @param:JsonProperty("sourceText") val sourceText: String,
    )

/**
 * Normalized text unit used for retrieval.
 *
 * @property id Text unit identifier.
 * @property chunkId Original chunk identifier.
 * @property text Text content.
 * @property sourcePath Source document path.
 */
@Serializable
data class TextUnit(
    val id: String,
    val chunkId: String,
    val text: String,
    val sourcePath: String,
)

/**
 * Summary text for an entity.
 *
 * @property entityId Entity identifier.
 * @property summary Summary text.
 */
@Serializable
data class EntitySummary(
    val entityId: String,
    val summary: String,
)

/**
 * Result of graph extraction, containing entities and relationships.
 *
 * @property entities Extracted entities.
 * @property relationships Extracted relationships.
 */
@Serializable
data class GraphExtractResult(
    val entities: List<Entity>,
    val relationships: List<Relationship>,
)
