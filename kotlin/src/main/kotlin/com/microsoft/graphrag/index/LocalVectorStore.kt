package com.microsoft.graphrag.index

import io.github.oshai.kotlinlogging.KotlinLogging
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
    private val payloadOverride: Payload? = null,
) {
    private val json = Json { prettyPrint = true }
    private val logger = KotlinLogging.logger {}

    fun save(
        textEmbeddings: List<TextEmbedding>,
        entityEmbeddings: List<EntityEmbedding>,
    ) {
        ensureParent(path)
        val payload = Payload(textEmbeddings, entityEmbeddings)
        Files.writeString(path, json.encodeToString(payload))
    }

    fun load(): Payload? {
        payloadOverride?.let { return it }
        if (!Files.exists(path)) return null
        return runCatching {
            json.decodeFromString<Payload>(Files.readString(path))
        }.getOrElse { error ->
            logger.warn { "Failed to load vector store from $path ($error); ignoring stored vectors." }
            null
        }
    }

    fun nearestEntities(
        query: List<Double>,
        limit: Int = 5,
    ): List<Pair<String, Double>> {
        val payload = load() ?: return emptyList()
        return linearSearch(payload.entityEmbeddings.map { it.entityId to it.vector }, query, limit)
    }

    fun nearestTextChunks(
        query: List<Double>,
        limit: Int = 5,
    ): List<Pair<String, Double>> {
        val payload = load() ?: return emptyList()
        return linearSearch(payload.textEmbeddings.map { it.chunkId to it.vector }, query, limit)
    }

    private fun ensureParent(p: Path) {
        val parent = p.parent ?: return
        if (!Files.exists(parent)) {
            Files.createDirectories(parent)
        }
    }

    private fun linearSearch(
        embeddings: List<Pair<String, List<Double>>>,
        query: List<Double>,
        k: Int,
    ): List<Pair<String, Double>> {
        if (embeddings.isEmpty() || query.isEmpty()) return emptyList()
        val vectors = embeddings.map { it.second.toDoubleArray() }.toTypedArray()
        val labels = embeddings.map { it.first }.toTypedArray()
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
