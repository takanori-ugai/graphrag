package com.microsoft.graphrag.cli

import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.nio.file.Files
import java.nio.file.Path
import java.time.Instant

@Serializable
sealed interface StorageValue {
    @Serializable
    data class Text(
        val value: String,
    ) : StorageValue

    @Serializable
    data class StringList(
        val value: List<String>,
    ) : StorageValue
}

@Serializable
data class StorageSnapshot(
    val metadata: Map<String, String>,
    val artifacts: Map<String, StorageValue>,
)

data class IndexPipelineResult(
    val storage: MemoryStorage,
    val metadata: Map<String, String>,
)

/**
 * Simplified in-memory storage for placeholder index artifacts.
 */
class MemoryStorage {
    private val data: MutableMap<String, StorageValue> = linkedMapOf()

    fun writeText(
        key: String,
        value: String,
    ) {
        data[key] = StorageValue.Text(value)
    }

    fun writeStringList(
        key: String,
        value: List<String>,
    ) {
        data[key] = StorageValue.StringList(value)
    }

    fun read(key: String): StorageValue? = data[key]

    fun describeKeys(): List<String> = data.keys.toList()

    fun snapshot(metadata: Map<String, String>): StorageSnapshot = StorageSnapshot(metadata, data.toMap())
}

/**
 * Placeholder index pipeline that pretends to run the indexing steps and writes synthetic
 * artifacts into a MemoryStorage instance. Intended to be replaced with the real Kotlin pipeline.
 */
class MemoryIndexPipeline(
    private val clock: () -> Instant = { Instant.now() },
) {
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
        val json =
            Json {
                prettyPrint = true
                encodeDefaults = true
            }.encodeToString(snapshot)
        Files.writeString(outputPath, json)
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
