package com.microsoft.graphrag.index

import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonIgnoreProperties
import com.fasterxml.jackson.annotation.JsonProperty
import kotlinx.serialization.Serializable

@Serializable
data class Covariate(
    val id: String,
    val subjectId: String,
    val covariateType: String = "covariate",
    val attributes: Map<String, String> = emptyMap(),
)

@Serializable
data class DocumentChunk(
    val id: String,
    val sourcePath: String,
    val text: String,
)

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

@Serializable
data class TextEmbedding(
    val chunkId: String,
    val vector: List<Double>,
)

@Serializable
data class EntityEmbedding(
    val entityId: String,
    val vector: List<Double>,
)

@Serializable
data class CommunityReportEmbedding(
    val communityId: Int,
    val vector: List<Double>,
)

@Serializable
data class CommunityAssignment(
    val entityId: String,
    val communityId: Int,
)

@Serializable
data class CommunityDetectionResult(
    val assignments: List<CommunityAssignment>,
    val hierarchy: Map<Int, Int>,
)

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

// Both kotlinx (for internal encoding) and Jackson (for LangChain4j responses) are needed here.
@Serializable
@JsonIgnoreProperties(ignoreUnknown = true)
data class Claim
    @JsonCreator
    constructor(
        @JsonProperty("subject") val subject: String,
        @JsonProperty("object") val `object`: String,
        @JsonProperty("claimType") val claimType: String,
        @JsonProperty("status") val status: String,
        @JsonProperty("startDate") val startDate: String,
        @JsonProperty("endDate") val endDate: String,
        @JsonProperty("description") val description: String,
        @JsonProperty("sourceText") val sourceText: String,
    )

@Serializable
data class TextUnit(
    val id: String,
    val chunkId: String,
    val text: String,
    val sourcePath: String,
)

@Serializable
data class EntitySummary(
    val entityId: String,
    val summary: String,
)

@Serializable
data class GraphExtractResult(
    val entities: List<Entity>,
    val relationships: List<Relationship>,
)
