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

data class TextEmbedding(
    val chunkId: String,
    val vector: List<Double>,
)

data class EntityEmbedding(
    val entityId: String,
    val vector: List<Double>,
)

data class GraphExtractResult(
    val entities: List<Entity>,
    val relationships: List<Relationship>,
)
