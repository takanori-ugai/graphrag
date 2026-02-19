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

        return IndexConfig(
            graphConfig =
                GraphRagConfig(
                    rootDir = configRoot,
                    inputDir = inputDir,
                    outputDir = outputDir,
                    updateOutputDir = updateOutputDir,
                ),
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
