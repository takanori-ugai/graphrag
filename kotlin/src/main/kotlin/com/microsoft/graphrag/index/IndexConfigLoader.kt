package com.microsoft.graphrag.index

import com.fasterxml.jackson.annotation.JsonProperty
import com.fasterxml.jackson.databind.DeserializationFeature
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import io.github.oshai.kotlinlogging.KotlinLogging
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path

/**
 * Resolved index configuration for the application.
 *
 * @property graphConfig GraphRag settings derived from the resolved paths.
 */
data class IndexConfig(
    val graphConfig: GraphRagConfig,
)

/**
 * Loads index configuration from a YAML file under the provided root.
 */
object IndexConfigLoader {
    private val logger = KotlinLogging.logger {}

    private val mapper: ObjectMapper =
        ObjectMapper(YAMLFactory())
            .registerKotlinModule()
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)

    /**
     * Loads configuration from `settings.yaml` (or an explicit config path) and resolves directories.
     *
     * @param root Project root used to resolve relative paths.
     * @param configPath Optional explicit config file path; defaults to `root/settings.yaml`.
     * @param overrideOutputDir Optional explicit output directory override.
     * @return IndexConfig containing a GraphRagConfig with resolved `rootDir`, `inputDir`, `outputDir`, and `updateOutputDir`.
     */
    fun load(
        root: Path,
        configPath: Path?,
        overrideOutputDir: Path? = null,
    ): IndexConfig {
        val resolvedRoot = root.toAbsolutePath().normalize()
        val settingsPath = (configPath ?: resolvedRoot.resolve("settings.yaml")).toAbsolutePath().normalize()

        val raw =
            if (Files.exists(settingsPath)) {
                try {
                    mapper.readValue(settingsPath.toFile(), RawIndexConfig::class.java)
                } catch (e: IOException) {
                    logger.warn { "Failed to parse $settingsPath ($e); using defaults." }
                    RawIndexConfig()
                }
            } else {
                if (configPath != null) {
                    logger.warn { "Config file not found: $settingsPath; using defaults." }
                }
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
        val path =
            if (value.isBlank()) {
                root.resolve(fallback)
            } else {
                val candidate = Path.of(value)
                if (candidate.isAbsolute) candidate else root.resolve(candidate)
            }
        return path.normalize()
    }
}

/**
 * Raw YAML-backed configuration container.
 *
 * @property rootDir Optional `root_dir` value; if absent, defaults to the provided root.
 * @property input Optional `input` configuration section.
 * @property output Optional `output` configuration section.
 */
data class RawIndexConfig(
    @param:JsonProperty("root_dir") val rootDir: String? = null,
    @param:JsonProperty("input") val input: RawInput? = null,
    @param:JsonProperty("output") val output: RawOutput? = null,
)

/**
 * Raw input configuration section.
 *
 * @property baseDir Optional `base_dir` override for input resolution.
 * @property storage Optional nested `storage` section.
 */
data class RawInput(
    @param:JsonProperty("base_dir") val baseDir: String? = null,
    @param:JsonProperty("storage") val storage: RawStorage? = null,
)

/**
 * Raw output configuration section.
 *
 * @property baseDir Optional `base_dir` override for output resolution.
 */
data class RawOutput(
    @param:JsonProperty("base_dir") val baseDir: String? = null,
)

/**
 * Raw storage configuration section.
 *
 * @property baseDir Optional `base_dir` override for storage input resolution.
 */
data class RawStorage(
    @param:JsonProperty("base_dir") val baseDir: String? = null,
)
