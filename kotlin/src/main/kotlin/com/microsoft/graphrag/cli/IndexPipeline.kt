package com.microsoft.graphrag.cli

import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.nio.file.Files
import java.nio.file.Path
import java.time.Instant

/**
 * Serializable representation of values stored by the in-memory pipeline.
 */
@Serializable
sealed interface StorageValue {
    /**
     * Textual storage value.
     *
     * @property value Stored text content.
     */
    @Serializable
    data class Text(
        val value: String,
    ) : StorageValue

    /**
     * Storage value containing a list of strings.
     *
     * @property value Stored string list.
     */
    @Serializable
    data class StringList(
        val value: List<String>,
    ) : StorageValue
}

/**
 * Serialized snapshot of memory storage and associated metadata.
 *
 * @property metadata Key/value metadata describing the pipeline run.
 * @property artifacts Stored artifacts keyed by name.
 */
@Serializable
data class StorageSnapshot(
    val metadata: Map<String, String>,
    val artifacts: Map<String, StorageValue>,
)

/**
 * Result produced by running the in-memory index pipeline.
 *
 * @property storage In-memory artifact store populated by the pipeline.
 * @property metadata Run metadata derived from options and inputs.
 */
data class IndexPipelineResult(
    val storage: MemoryStorage,
    val metadata: Map<String, String>,
)

/**
 * Simplified in-memory storage for placeholder index artifacts.
 */
class MemoryStorage {
    private val data: MutableMap<String, StorageValue> = linkedMapOf()

    /**
     * Writes a text artifact into storage.
     *
     * @param key Artifact key.
     * @param value Text content to store.
     */
    fun writeText(
        key: String,
        value: String,
    ) {
        data[key] = StorageValue.Text(value)
    }

    /**
     * Writes a string list artifact into storage.
     *
     * @param key Artifact key.
     * @param value String list content to store.
     */
    fun writeStringList(
        key: String,
        value: List<String>,
    ) {
        data[key] = StorageValue.StringList(value)
    }

    /**
     * Reads a stored artifact by key.
     *
     * @param key Artifact key to look up.
     * @return The stored value, or null if absent.
     */
    fun read(key: String): StorageValue? = data[key]

    /**
     * Lists all stored artifact keys in insertion order.
     *
     * @return Keys currently stored.
     */
    fun describeKeys(): List<String> = data.keys.toList()

    /**
     * Builds a serializable snapshot of storage and metadata.
     *
     * @param metadata Metadata to include in the snapshot.
     * @return Serialized storage snapshot.
     */
    fun snapshot(metadata: Map<String, String>): StorageSnapshot = StorageSnapshot(metadata, data.toMap())
}

/**
 * Placeholder index pipeline that writes synthetic artifacts into memory storage.
 *
 * @property clock Clock supplier used to timestamp output.
 */
class MemoryIndexPipeline(
    private val clock: () -> Instant = { Instant.now() },
) {
    private val json =
        Json {
            prettyPrint = true
            encodeDefaults = true
        }

    /**
     * Executes the pipeline for the provided options.
     *
     * @param options Indexing options for the run.
     * @return A result containing the in-memory storage and metadata.
     */
    fun run(options: IndexOptions): IndexPipelineResult {
        val storage = MemoryStorage()

        val inputDocs = discoverInput(options.root.resolve("input"))
        storage.writeStringList("input_docs", inputDocs)

        val configPath = options.config ?: options.root.resolve("settings.yaml")
        storage.writeText("config_path", configPath.toAbsolutePath().normalize().toString())

        storage.writeText(
            "summary",
            "Placeholder index summary for root=${options.root} method=${options.method} docs=${inputDocs.size}",
        )
        storage.writeText("timestamp", clock().toString())

        val metadata =
            mapOf(
                "method" to options.method,
                "root" to options.root.toString(),
                "update" to options.isUpdate.toString(),
                "output" to (options.output?.toString() ?: "in-memory"),
                "documentsIndexed" to inputDocs.size.toString(),
            )

        return IndexPipelineResult(storage = storage, metadata = metadata)
    }

    /**
     * Persists an in-memory pipeline result as a JSON snapshot.
     *
     * @param result Pipeline result to persist.
     * @param outputDir Directory where the JSON snapshot is written.
     * @return Path to the written JSON file.
     */
    fun persistToJson(
        result: IndexPipelineResult,
        outputDir: Path,
    ): Path {
        if (!Files.exists(outputDir)) {
            Files.createDirectories(outputDir)
        }
        val fileName = if (result.metadata["update"] == "true") "update_artifacts.json" else "index_artifacts.json"
        val outputPath = outputDir.resolve(fileName)
        val snapshot = result.storage.snapshot(result.metadata)
        val payload = json.encodeToString(snapshot)
        Files.writeString(outputPath, payload)
        return outputPath
    }

    private fun discoverInput(inputDir: Path): List<String> {
        if (!Files.exists(inputDir)) {
            return emptyList()
        }
        return Files.walk(inputDir).use { walk ->
            walk
                .filter { path -> Files.isRegularFile(path) }
                .map { it.toString() }
                .toList()
        }
    }
}
