package com.microsoft.graphrag.index

import com.fasterxml.jackson.annotation.JsonIgnoreProperties
import com.fasterxml.jackson.annotation.JsonProperty
import com.fasterxml.jackson.databind.DeserializationFeature
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import io.github.oshai.kotlinlogging.KotlinLogging
import java.nio.file.Files
import java.nio.file.Path

data class IndexConfig(
    val graphConfig: GraphRagConfig,
)

object IndexConfigLoader {
    private val logger = KotlinLogging.logger {}

    private val mapper: ObjectMapper =
        ObjectMapper(YAMLFactory())
            .registerKotlinModule()
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)

    fun load(
        root: Path,
        configPath: Path?,
        overrideOutputDir: Path? = null,
    ): IndexConfig {
        val resolvedRoot = root.toAbsolutePath().normalize()
        val settingsPath = (configPath ?: resolvedRoot.resolve("settings.yaml")).toAbsolutePath().normalize()

        val raw =
            if (Files.exists(settingsPath)) {
                runCatching { mapper.readValue(settingsPath.toFile(), RawIndexConfig::class.java) }.getOrElse { error ->
                    logger.warn { "Failed to parse $settingsPath ($error); using defaults." }
                    RawIndexConfig()
                }
            } else {
                RawIndexConfig()
            }

        val configRoot = raw.rootDir?.let { resolvedRoot.resolve(it).normalize() } ?: resolvedRoot
        val inputDir =
            resolveBaseDir(
                configRoot,
                raw.input?.storage?.baseDir ?: raw.input?.baseDir,
                "input",
            )

        val outputDir =
            overrideOutputDir?.toAbsolutePath()?.normalize()
                ?: resolveBaseDir(
                    configRoot,
                    raw.output?.baseDir,
                    "output",
                )

        val updateOutputDir = outputDir.resolve("update_output")

        val extractOptions = buildExtractGraphOptions(configRoot, raw.extractGraph)

        return IndexConfig(
            graphConfig =
                GraphRagConfig(
                    rootDir = configRoot,
                    inputDir = inputDir,
                    outputDir = outputDir,
                    updateOutputDir = updateOutputDir,
                    extractGraphOptions = extractOptions,
                ),
        )
    }

    private fun buildExtractGraphOptions(
        root: Path,
        raw: RawExtractGraph?,
    ): ExtractGraphOptions {
        if (raw == null) return ExtractGraphOptions()

        val promptOverride =
            raw.prompt?.let { promptPath ->
                val resolved = root.resolve(promptPath).normalize()
                if (Files.exists(resolved)) {
                    Files.readString(resolved)
                } else {
                    logger.warn { "Extract graph prompt not found at $resolved; using default prompt." }
                    null
                }
            }

        return ExtractGraphOptions(
            entityTypes = raw.entityTypes ?: ExtractGraphWorkflow.DEFAULT_ENTITY_TYPES,
            maxConcurrency = raw.maxConcurrency ?: 4,
            useCache = raw.useCache ?: true,
            cacheKeyPrefix = raw.cacheKeyPrefix ?: "extract_graph",
            promptOverride = promptOverride,
        )
    }

    private fun resolveBaseDir(
        root: Path,
        configured: String?,
        fallback: String,
    ): Path {
        val value = configured?.trim().orEmpty()
        if (value.isBlank()) return root.resolve(fallback).normalize()
        val candidate = Path.of(value)
        return if (candidate.isAbsolute) candidate.normalize() else root.resolve(candidate).normalize()
    }
}

@JsonIgnoreProperties(ignoreUnknown = true)
data class RawIndexConfig(
    @param:JsonProperty("root_dir") val rootDir: String? = null,
    @param:JsonProperty("input") val input: RawInput? = null,
    @param:JsonProperty("output") val output: RawOutput? = null,
    @param:JsonProperty("extract_graph") val extractGraph: RawExtractGraph? = null,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class RawInput(
    @param:JsonProperty("base_dir") val baseDir: String? = null,
    @param:JsonProperty("storage") val storage: RawStorage? = null,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class RawOutput(
    @param:JsonProperty("base_dir") val baseDir: String? = null,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class RawStorage(
    @param:JsonProperty("base_dir") val baseDir: String? = null,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class RawExtractGraph(
    @param:JsonProperty("prompt") val prompt: String? = null,
    @param:JsonProperty("entity_types") val entityTypes: List<String>? = null,
    @param:JsonProperty("max_concurrency") val maxConcurrency: Int? = null,
    @param:JsonProperty("use_cache") val useCache: Boolean? = null,
    @param:JsonProperty("cache_key_prefix") val cacheKeyPrefix: String? = null,
)
