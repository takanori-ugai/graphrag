package com.microsoft.graphrag.index

import kotlinx.serialization.Serializable

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
)

@Serializable
data class Relationship(
    val sourceId: String,
    val targetId: String,
    val type: String,
    val description: String? = null,
    val sourceChunkId: String,
)

@kotlinx.serialization.Serializable
data class TextEmbedding(
    val chunkId: String,
    val vector: List<Double>,
)

@kotlinx.serialization.Serializable
data class EntityEmbedding(
    val entityId: String,
    val vector: List<Double>,
)

@kotlinx.serialization.Serializable
data class CommunityAssignment(
    val entityId: String,
    val communityId: Int,
)

data class CommunityDetectionResult(
    val assignments: List<CommunityAssignment>,
    val hierarchy: Map<Int, Int>,
)

@kotlinx.serialization.Serializable
data class CommunityReport(
    val communityId: Int,
    val summary: String,
    val parentCommunityId: Int? = null,
)

@kotlinx.serialization.Serializable
data class Claim(
    val subject: String,
    val `object`: String,
    val claimType: String,
    val status: String,
    val startDate: String,
    val endDate: String,
    val description: String,
    val sourceText: String,
)

@kotlinx.serialization.Serializable
data class TextUnit(
    val id: String,
    val chunkId: String,
    val text: String,
    val sourcePath: String,
)

@kotlinx.serialization.Serializable
data class EntitySummary(
    val entityId: String,
    val summary: String,
)

data class GraphExtractResult(
    val entities: List<Entity>,
    val relationships: List<Relationship>,
)
