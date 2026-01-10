package com.microsoft.graphrag.index

import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import smile.math.distance.EuclideanDistance
import smile.neighbor.LinearSearch
import java.nio.file.Files
import java.nio.file.Path

class LocalVectorStore(
    private val path: Path,
) {
    private val json = Json { prettyPrint = true }

    fun save(
        textEmbeddings: List<TextEmbedding>,
        entityEmbeddings: List<EntityEmbedding>,
    ) {
        ensureParent(path)
        val payload = Payload(textEmbeddings, entityEmbeddings)
        Files.writeString(path, json.encodeToString(payload))
    }

    fun load(): Payload? {
        if (!Files.exists(path)) return null
        return runCatching {
            json.decodeFromString<Payload>(Files.readString(path))
        }.getOrNull()
    }

    fun nearestEntities(
        query: List<Double>,
        limit: Int = 5,
    ): List<Pair<String, Double>> {
        val payload = load() ?: return emptyList()
        return linearSearch(payload.entityEmbeddings, query, limit)
    }

    private fun ensureParent(p: Path) {
        val parent = p.parent ?: return
        if (!Files.exists(parent)) {
            Files.createDirectories(parent)
        }
    }

    private fun linearSearch(
        embeddings: List<EntityEmbedding>,
        query: List<Double>,
        k: Int,
    ): List<Pair<String, Double>> {
        if (embeddings.isEmpty() || query.isEmpty()) return emptyList()
        val vectors = embeddings.map { it.vector.toDoubleArray() }.toTypedArray()
        val labels = embeddings.map { it.entityId }.toTypedArray()
        val search = LinearSearch(vectors, labels, EuclideanDistance())
        val q = query.toDoubleArray()
        val limit = k.coerceAtMost(vectors.size)
        return search.search(q, limit).map { it.value to it.distance }
    }
}

@Serializable
data class Payload(
    val textEmbeddings: List<TextEmbedding>,
    val entityEmbeddings: List<EntityEmbedding>,
)
