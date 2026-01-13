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

    /**
     * Writes the provided text and entity embeddings to the store file as a JSON payload, creating the parent directory if necessary.
     *
     * @param textEmbeddings The list of text embeddings to persist.
     * @param entityEmbeddings The list of entity embeddings to persist.
     */
    fun save(
        textEmbeddings: List<TextEmbedding>,
        entityEmbeddings: List<EntityEmbedding>,
    ) {
        ensureParent(path)
        val payload = Payload(textEmbeddings, entityEmbeddings)
        Files.writeString(path, json.encodeToString(payload))
    }

    /**
     * Load the stored Payload from the configured path or return the configured override if present.
     *
     * @return `Payload` loaded from disk, the configured override if provided, or `null` if the file does not exist or cannot be deserialized.
     */
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

    /**
     * Finds the nearest entities to a query embedding.
     *
     * @param query The query embedding vector to compare against stored entity vectors.
     * @param limit Maximum number of nearest entities to return.
     * @return A list of pairs where each pair is the entity ID and its Euclidean distance to the query, ordered from nearest to farthest.
     */
    fun nearestEntities(
        query: List<Double>,
        limit: Int = 5,
    ): List<Pair<String, Double>> {
        val payload = load() ?: return emptyList()
        return linearSearch(payload.entityEmbeddings.map { it.entityId to it.vector }, query, limit)
    }

    /**
     * Finds the nearest stored text chunks to the provided query embedding.
     *
     * @param query The query embedding vector to match against stored chunk vectors.
     * @param limit Maximum number of nearest chunks to return.
     * @return A list of pairs where each pair contains a chunk ID and its distance to the query; pairs are ordered from nearest to farthest.
     */
    fun nearestTextChunks(
        query: List<Double>,
        limit: Int = 5,
    ): List<Pair<String, Double>> {
        val payload = load() ?: return emptyList()
        return linearSearch(payload.textEmbeddings.map { it.chunkId to it.vector }, query, limit)
    }

    /**
     * Ensures the parent directory of the given path exists, creating it (and any missing ancestors) if necessary.
     *
     * @param p The path whose parent directory should be created when missing.
     */
    private fun ensureParent(p: Path) {
        val parent = p.parent ?: return
        if (!Files.exists(parent)) {
            Files.createDirectories(parent)
        }
    }

    /**
     * Finds up to `k` nearest neighbors to `query` among `embeddings` using Euclidean distance.
     *
     * @param embeddings List of pairs where the first element is the label (id) and the second is the embedding vector.
     * @param query The query embedding vector to search for.
     * @param k Maximum number of nearest neighbors to return.
     * @return A list of up to `k` pairs `(label, distance)`, where `distance` is the Euclidean distance between the query and the embedding.
     *         Returns an empty list if `embeddings` or `query` is empty.
     */
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
