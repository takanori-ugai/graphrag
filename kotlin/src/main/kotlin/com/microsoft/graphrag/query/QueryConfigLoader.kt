package com.microsoft.graphrag.query

import com.fasterxml.jackson.annotation.JsonIgnoreProperties
import com.fasterxml.jackson.annotation.JsonProperty
import com.fasterxml.jackson.databind.DeserializationFeature
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import java.nio.file.Files
import java.nio.file.Path

data class QueryConfig(
    val root: Path,
    val outputDirs: List<Path>,
    val chatModel: String? = null,
    val embeddingModel: String? = null,
)

object QueryConfigLoader {
    private val mapper: ObjectMapper =
        ObjectMapper(YAMLFactory())
            .registerKotlinModule()
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)

    fun load(
        root: Path,
        configPath: Path?,
        overrideOutputDirs: List<Path> = emptyList(),
    ): QueryConfig {
        val resolvedRoot = root.toAbsolutePath().normalize()
        val settingsPath = (configPath ?: resolvedRoot.resolve("settings.yaml")).toAbsolutePath().normalize()
        if (!Files.exists(settingsPath)) {
            val outputs =
                if (overrideOutputDirs.isNotEmpty()) {
                    overrideOutputDirs
                } else {
                    listOf(resolvedRoot.resolve("sample-index/output"))
                }
            return QueryConfig(root = resolvedRoot, outputDirs = outputs)
        }

        val raw =
            runCatching {
                mapper.readValue(settingsPath.toFile(), RawConfig::class.java)
            }.getOrElse { error ->
                val outputs =
                    if (overrideOutputDirs.isNotEmpty()) {
                        overrideOutputDirs
                    } else {
                        listOf(resolvedRoot.resolve("output"))
                    }
                return QueryConfig(root = resolvedRoot, outputDirs = outputs).also {
                    println("Warning: failed to parse $settingsPath ($error); using default outputs.")
                }
            }
        val configRoot = raw.rootDir?.let { resolvedRoot.resolve(it).normalize() } ?: resolvedRoot
        val outputDirs =
            when {
                overrideOutputDirs.isNotEmpty() -> {
                    overrideOutputDirs.map { it.toAbsolutePath().normalize() }
                }

                raw.outputs.isNotEmpty() -> {
                    raw.outputs.values.mapNotNull {
                        it.baseDir?.let { base ->
                            configRoot.resolve(base).normalize()
                        }
                    }
                }

                raw.output?.baseDir != null -> {
                    listOf(configRoot.resolve(raw.output.baseDir).normalize())
                }

                else -> {
                    listOf(configRoot.resolve("output"))
                }
            }

        return QueryConfig(
            root = configRoot,
            outputDirs = outputDirs.ifEmpty { listOf(configRoot.resolve("output")) },
            chatModel = raw.models["default_chat_model"]?.model,
            embeddingModel = raw.models["default_embedding_model"]?.model,
        )
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    private data class RawConfig(
        @JsonProperty("root_dir") val rootDir: String? = null,
        val output: RawStorageConfig? = null,
        val outputs: Map<String, RawStorageConfig> = emptyMap(),
        val models: Map<String, RawModelConfig> = emptyMap(),
    )

    @JsonIgnoreProperties(ignoreUnknown = true)
    private data class RawStorageConfig(
        @JsonProperty("base_dir") val baseDir: String? = null,
    )

    @JsonIgnoreProperties(ignoreUnknown = true)
    private data class RawModelConfig(
        val model: String? = null,
    )
}
