package com.microsoft.graphrag.query

import com.fasterxml.jackson.annotation.JsonIgnoreProperties
import com.fasterxml.jackson.annotation.JsonProperty
import com.fasterxml.jackson.databind.DeserializationFeature
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import com.microsoft.graphrag.prompts.query.QUESTION_SYSTEM_PROMPT
import com.microsoft.graphrag.query.DriftSearchEngine.Companion.DEFAULT_DRIFT_PRIMER_PROMPT
import com.microsoft.graphrag.query.DriftSearchEngine.Companion.DEFAULT_DRIFT_REDUCE_PROMPT
import com.microsoft.graphrag.query.GlobalSearchEngine.Companion.DEFAULT_GENERAL_KNOWLEDGE_INSTRUCTION
import com.microsoft.graphrag.query.GlobalSearchEngine.Companion.DEFAULT_MAP_SYSTEM_PROMPT
import com.microsoft.graphrag.query.GlobalSearchEngine.Companion.DEFAULT_REDUCE_SYSTEM_PROMPT
import io.github.oshai.kotlinlogging.KotlinLogging
import java.nio.file.Files
import java.nio.file.Path

data class QueryConfig(
    val root: Path,
    val indexes: List<QueryIndexConfig>,
    val defaultChatModel: QueryModelConfig?,
    val defaultEmbeddingModel: String?,
    val basic: BasicSearchConfig,
    val local: LocalSearchConfig,
    val global: GlobalSearchConfig,
    val drift: DriftSearchConfig,
    val questionPrompt: String,
)

data class QueryIndexConfig(
    val name: String,
    val path: Path,
)

data class QueryModelConfig(
    val model: String?,
    val params: ModelParams = ModelParams(),
)

data class BasicSearchConfig(
    val prompt: String,
    val chat: QueryModelConfig?,
    val embeddingModel: String?,
    val k: Int,
    val maxContextTokens: Int,
)

data class LocalSearchConfig(
    val prompt: String,
    val chat: QueryModelConfig?,
    val embeddingModel: String?,
    val textUnitProp: Double,
    val communityProp: Double,
    val conversationHistoryMaxTurns: Int,
    val topKEntities: Int,
    val topKRelationships: Int,
    val maxContextTokens: Int,
)

data class DynamicCommunityConfig(
    val threshold: Int,
    val keepParent: Boolean,
    val numRepeats: Int,
    val useSummary: Boolean,
    val maxLevel: Int,
)

data class GlobalSearchConfig(
    val mapPrompt: String,
    val reducePrompt: String,
    val knowledgePrompt: String?,
    val chat: QueryModelConfig?,
    val embeddingModel: String?,
    val allowGeneralKnowledge: Boolean,
    val mapMaxLength: Int,
    val reduceMaxLength: Int,
    val maxContextTokens: Int,
    val dataMaxTokens: Int,
    val dynamic: DynamicCommunityConfig,
)

data class DriftSearchConfig(
    val prompt: String,
    val reducePrompt: String,
    val chat: QueryModelConfig?,
    val embeddingModel: String?,
    val maxIterations: Int,
)

object QueryConfigLoader {
    private const val DEFAULT_CHAT_ID = "default_chat_model"
    private const val DEFAULT_EMBEDDING_ID = "default_embedding_model"
    private val logger = KotlinLogging.logger {}

    private val mapper: ObjectMapper =
        ObjectMapper(YAMLFactory())
            .registerKotlinModule()
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)

    /**
     * Builds a QueryConfig for the given project root by loading and merging configuration from a settings file.
     *
     * If a settings file is provided (via `configPath`) or found at `<root>/settings.yaml`, that file is parsed
     * and used
     * to populate index definitions, default model selections, prompts, and per-component search settings. If the file
     * is missing or cannot be parsed, a sensible default configuration is returned based on `root` and any provided
     * `overrideOutputDirs`.
     *
     * @param root The filesystem root path for the project; used to resolve relative paths in the configuration.
     * @param configPath Optional explicit path to a settings file; when null the loader looks for
     * `settings.yaml` under `root`.
     * @param overrideOutputDirs Optional list of output directories that override outputs declared in the
     * settings file.
     * @return A fully populated QueryConfig containing resolved root, index configurations, default
     * chat/embedding models,
     * and the assembled BasicSearchConfig, LocalSearchConfig, GlobalSearchConfig, DriftSearchConfig, and
     * question prompt.
     */
    fun load(
        root: Path,
        configPath: Path?,
        overrideOutputDirs: List<Path> = emptyList(),
    ): QueryConfig {
        val resolvedRoot = root.toAbsolutePath().normalize()
        val settingsPath = (configPath ?: resolvedRoot.resolve("settings.yaml")).toAbsolutePath().normalize()
        val overrideDirs = overrideOutputDirs.map { it.toAbsolutePath().normalize() }

        if (!Files.exists(settingsPath)) {
            val indexes = buildIndexConfigs(resolvedRoot, null, overrideDirs)
            return defaultConfig(resolvedRoot, indexes)
        }

        val raw =
            runCatching {
                mapper.readValue(settingsPath.toFile(), RawConfig::class.java)
            }.getOrElse { error ->
                logger.warn { "Failed to parse $settingsPath ($error); using default outputs." }
                val indexes = buildIndexConfigs(resolvedRoot, null, overrideDirs)
                return defaultConfig(resolvedRoot, indexes)
            }

        val configRoot = raw.rootDir?.let { resolvedRoot.resolve(it).normalize() } ?: resolvedRoot
        val indexes = buildIndexConfigs(configRoot, raw, overrideDirs)

        val defaultChat = resolveModel(raw.models, DEFAULT_CHAT_ID)
        val defaultEmbedding = resolveEmbedding(raw, DEFAULT_EMBEDDING_ID)

        val basicPrompt = loadPrompt(configRoot, raw.basicSearch?.prompt, DEFAULT_BASIC_SEARCH_SYSTEM_PROMPT)
        val localPrompt = loadPrompt(configRoot, raw.localSearch?.prompt, DEFAULT_LOCAL_SEARCH_SYSTEM_PROMPT)
        val globalMapPrompt = loadPrompt(configRoot, raw.globalSearch?.mapPrompt, DEFAULT_MAP_SYSTEM_PROMPT)
        val globalReducePrompt = loadPrompt(configRoot, raw.globalSearch?.reducePrompt, DEFAULT_REDUCE_SYSTEM_PROMPT)
        val knowledgePrompt = raw.globalSearch?.knowledgePrompt?.let { loadPrompt(configRoot, it, DEFAULT_GENERAL_KNOWLEDGE_INSTRUCTION) }
        val driftPrompt = loadPrompt(configRoot, raw.driftSearch?.prompt, DEFAULT_DRIFT_PRIMER_PROMPT)
        val driftReducePrompt = loadPrompt(configRoot, raw.driftSearch?.reducePrompt, DEFAULT_DRIFT_REDUCE_PROMPT)
        val questionPrompt = loadPrompt(configRoot, raw.questionGenPrompt, QUESTION_SYSTEM_PROMPT)

        val basic =
            BasicSearchConfig(
                prompt = basicPrompt,
                chat = resolveModel(raw.models, raw.basicSearch?.chatModelId) ?: defaultChat,
                embeddingModel = resolveEmbedding(raw, raw.basicSearch?.embeddingModelId) ?: defaultEmbedding,
                k = raw.basicSearch?.k ?: 10,
                maxContextTokens = raw.basicSearch?.maxContextTokens ?: 12_000,
            )

        val local =
            LocalSearchConfig(
                prompt = localPrompt,
                chat = resolveModel(raw.models, raw.localSearch?.chatModelId) ?: defaultChat,
                embeddingModel = resolveEmbedding(raw, raw.localSearch?.embeddingModelId) ?: defaultEmbedding,
                textUnitProp = raw.localSearch?.textUnitProp ?: 0.5,
                communityProp = raw.localSearch?.communityProp ?: 0.15,
                conversationHistoryMaxTurns = raw.localSearch?.conversationHistoryMaxTurns ?: 5,
                topKEntities = raw.localSearch?.topKEntities ?: 10,
                topKRelationships = raw.localSearch?.topKRelationships ?: 10,
                maxContextTokens = raw.localSearch?.maxContextTokens ?: 12_000,
            )

        val dynamic =
            DynamicCommunityConfig(
                threshold = raw.globalSearch?.dynamicSearchThreshold ?: 1,
                keepParent = raw.globalSearch?.dynamicSearchKeepParent ?: false,
                numRepeats = raw.globalSearch?.dynamicSearchNumRepeats ?: 1,
                useSummary = raw.globalSearch?.dynamicSearchUseSummary ?: false,
                maxLevel = raw.globalSearch?.dynamicSearchMaxLevel ?: 2,
            )

        val global =
            GlobalSearchConfig(
                mapPrompt = globalMapPrompt,
                reducePrompt = globalReducePrompt,
                knowledgePrompt = knowledgePrompt,
                chat = resolveModel(raw.models, raw.globalSearch?.chatModelId) ?: defaultChat,
                embeddingModel = resolveEmbedding(raw, raw.globalSearch?.embeddingModelId) ?: defaultEmbedding,
                allowGeneralKnowledge = raw.globalSearch?.knowledgePrompt != null,
                mapMaxLength = raw.globalSearch?.mapMaxLength ?: 1_000,
                reduceMaxLength = raw.globalSearch?.reduceMaxLength ?: 2_000,
                maxContextTokens = raw.globalSearch?.maxContextTokens ?: 12_000,
                dataMaxTokens = raw.globalSearch?.dataMaxTokens ?: 12_000,
                dynamic = dynamic,
            )

        val drift =
            DriftSearchConfig(
                prompt = driftPrompt,
                reducePrompt = driftReducePrompt,
                chat = resolveModel(raw.models, raw.driftSearch?.chatModelId) ?: defaultChat,
                embeddingModel = resolveEmbedding(raw, raw.driftSearch?.embeddingModelId) ?: defaultEmbedding,
                maxIterations = raw.driftSearch?.nDepth ?: 3,
            )

        return QueryConfig(
            root = configRoot,
            indexes = indexes,
            defaultChatModel = defaultChat,
            defaultEmbeddingModel = defaultEmbedding,
            basic = basic,
            local = local,
            global = global,
            drift = drift,
            questionPrompt = questionPrompt,
        )
    }

    /**
     * Build a QueryConfig populated with sensible defaults derived from the given root and index configurations.
     *
     * If `indexConfigs` is empty, a single default index at `sample-index/output` under `root` is used. The returned
     * configuration includes default chat and embedding models and default settings for basic, local, global, and drift
     * search behaviors as well as the question generation prompt.
     *
     * @param root The base filesystem path used to resolve default index locations.
     * @param indexConfigs A list of index configurations to include; may be empty to trigger the default index
     * creation.
     * @return A fully populated QueryConfig with provided or default indexes and default model/search settings.
     */
    private fun defaultConfig(
        root: Path,
        indexConfigs: List<QueryIndexConfig>,
    ): QueryConfig {
        val indexes =
            indexConfigs.ifEmpty {
                listOf(QueryIndexConfig("default", root.resolve("sample-index/output")))
            }
        val defaultChat = QueryModelConfig("gpt-4o-mini")
        val defaultEmbedding = "text-embedding-3-small"
        return QueryConfig(
            root = root,
            indexes = indexes,
            defaultChatModel = defaultChat,
            defaultEmbeddingModel = defaultEmbedding,
            basic =
                BasicSearchConfig(
                    prompt = DEFAULT_BASIC_SEARCH_SYSTEM_PROMPT,
                    chat = defaultChat,
                    embeddingModel = defaultEmbedding,
                    k = 10,
                    maxContextTokens = 12_000,
                ),
            local =
                LocalSearchConfig(
                    prompt = DEFAULT_LOCAL_SEARCH_SYSTEM_PROMPT,
                    chat = defaultChat,
                    embeddingModel = defaultEmbedding,
                    textUnitProp = 0.5,
                    communityProp = 0.15,
                    conversationHistoryMaxTurns = 5,
                    topKEntities = 10,
                    topKRelationships = 10,
                    maxContextTokens = 12_000,
                ),
            global =
                GlobalSearchConfig(
                    mapPrompt = DEFAULT_MAP_SYSTEM_PROMPT,
                    reducePrompt = DEFAULT_REDUCE_SYSTEM_PROMPT,
                    knowledgePrompt = DEFAULT_GENERAL_KNOWLEDGE_INSTRUCTION,
                    chat = defaultChat,
                    embeddingModel = defaultEmbedding,
                    allowGeneralKnowledge = false,
                    mapMaxLength = 1_000,
                    reduceMaxLength = 2_000,
                    maxContextTokens = 12_000,
                    dataMaxTokens = 12_000,
                    dynamic =
                        DynamicCommunityConfig(
                            threshold = 1,
                            keepParent = false,
                            numRepeats = 1,
                            useSummary = false,
                            maxLevel = 2,
                        ),
                ),
            drift =
                DriftSearchConfig(
                    prompt = DEFAULT_DRIFT_PRIMER_PROMPT,
                    reducePrompt = DEFAULT_DRIFT_REDUCE_PROMPT,
                    chat = defaultChat,
                    embeddingModel = defaultEmbedding,
                    maxIterations = 3,
                ),
            questionPrompt = QUESTION_SYSTEM_PROMPT,
        )
    }

    /**
     * Constructs index configurations using override directories, explicit outputs from the raw config, or
     * sensible defaults.
     *
     * When `overrideDirs` is non-empty it maps each provided path to an index (names are taken from the raw outputs by
     * position when available, otherwise from the directory name). If `overrideDirs` is empty but `raw.outputs`
     * is present
     * those entries are used. If `raw.output.baseDir` is set a single index named "output" is created from that base
     * directory. Otherwise a default "output" index at `configRoot/sample-index/output` is returned.
     *
     * Names are sanitized and returned paths are normalized (and absolute for overrides).
     *
     * @param configRoot Base directory used to resolve relative output/baseDir values from the raw configuration.
     * @param raw Parsed raw configuration that may contain `outputs` (multiple) or `output.baseDir` (single).
     * @param overrideDirs If non-empty, these explicit directories take precedence and each becomes an index entry.
     * @return A list of QueryIndexConfig instances with sanitized names and normalized paths.
     */
    private fun buildIndexConfigs(
        configRoot: Path,
        raw: RawConfig?,
        overrideDirs: List<Path>,
    ): List<QueryIndexConfig> {
        if (overrideDirs.isNotEmpty()) {
            val namesFromConfig =
                raw
                    ?.outputs
                    ?.keys
                    ?.toList()
                    .orEmpty()
            return overrideDirs.mapIndexed { idx, path ->
                val derivedName = namesFromConfig.getOrNull(idx) ?: path.fileName?.toString().orEmpty()
                QueryIndexConfig(
                    name = sanitizeName(derivedName.ifBlank { "index${idx + 1}" }),
                    path = path.toAbsolutePath().normalize(),
                )
            }
        }

        if (raw != null && raw.outputs.isNotEmpty()) {
            return raw.outputs.map { (name, storage) ->
                QueryIndexConfig(
                    name = sanitizeName(name),
                    path = configRoot.resolve(storage.baseDir ?: "output").normalize(),
                )
            }
        }

        raw?.output?.baseDir?.let { base ->
            return listOf(QueryIndexConfig("output", configRoot.resolve(base).normalize()))
        }

        return listOf(QueryIndexConfig("output", configRoot.resolve("sample-index/output")))
    }

    /**
     * Resolve a raw model entry by id and construct a corresponding QueryModelConfig.
     *
     * @param models Map of model ids to their raw configurations.
     * @param id The model id to resolve; if `null` or not present in `models`, the function returns `null`.
     * @return `QueryModelConfig` for the resolved model, or `null` if no matching entry exists. The returned
     * model's `ModelParams.maxTokens` is `raw.maxTokens` if present, otherwise `raw.maxCompletionTokens`. */
    private fun resolveModel(
        models: Map<String, RawModelConfig>,
        id: String?,
    ): QueryModelConfig? {
        val modelConfig = id?.let { models[it] } ?: return null
        val params =
            ModelParams(
                temperature = modelConfig.temperature,
                topP = modelConfig.topP,
                maxTokens = modelConfig.maxTokens ?: modelConfig.maxCompletionTokens,
            )
        return QueryModelConfig(modelConfig.model, params)
    }

    /**
     * Resolve the embedding model name from the raw configuration, falling back to the default embedding id.
     *
     * @param raw RawConfig containing model definitions.
     * @param id Optional embedding model id to look up; when null the default embedding id is used.
     * @return The embedding model name for `id` if defined, otherwise the model name for the default embedding
     * id, or `null` if neither is found.
     */
    private fun resolveEmbedding(
        raw: RawConfig,
        id: String?,
    ): String? {
        val targetId = id ?: DEFAULT_EMBEDDING_ID
        return raw.models[targetId]?.model ?: raw.models[DEFAULT_EMBEDDING_ID]?.model
    }

    /**
     * Loads a prompt from a file path resolved against `root`, or returns the provided fallback.
     *
     * @param root Base directory used to resolve `path` when `path` is relative.
     * @param path File path to the prompt; may be relative to `root` or absolute. If `null` or blank,
     * `fallback` is returned.
     * @param fallback Default prompt text to return when `path` is missing or the file cannot be read.
     * @return The prompt file contents if successfully read, otherwise `fallback`.
     */
    private fun loadPrompt(
        root: Path,
        path: String?,
        fallback: String,
    ): String {
        if (path.isNullOrBlank()) return fallback
        val resolved =
            runCatching { root.resolve(path).normalize() }
                .getOrDefault(Path.of(path).toAbsolutePath().normalize())
        return runCatching { Files.readString(resolved) }.getOrElse { error ->
            logger.warn { "Failed to load prompt at $resolved ($error); using default." }
            fallback
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    private data class RawConfig(
        @JsonProperty("root_dir") val rootDir: String? = null,
        val output: RawStorageConfig? = null,
        val outputs: Map<String, RawStorageConfig> = emptyMap(),
        val models: Map<String, RawModelConfig> = emptyMap(),
        @JsonProperty("basic_search") val basicSearch: RawBasicSearch? = null,
        @JsonProperty("local_search") val localSearch: RawLocalSearch? = null,
        @JsonProperty("global_search") val globalSearch: RawGlobalSearch? = null,
        @JsonProperty("drift_search") val driftSearch: RawDriftSearch? = null,
        @JsonProperty("question_gen_prompt") val questionGenPrompt: String? = null,
    )

    @JsonIgnoreProperties(ignoreUnknown = true)
    private data class RawStorageConfig(
        @JsonProperty("base_dir") val baseDir: String? = null,
    )

    @JsonIgnoreProperties(ignoreUnknown = true)
    private data class RawModelConfig(
        val model: String? = null,
        val temperature: Double? = null,
        @JsonProperty("top_p") val topP: Double? = null,
        @JsonProperty("max_tokens") val maxTokens: Int? = null,
        @JsonProperty("max_completion_tokens") val maxCompletionTokens: Int? = null,
    )

    @JsonIgnoreProperties(ignoreUnknown = true)
    private data class RawBasicSearch(
        val prompt: String? = null,
        @JsonProperty("chat_model_id") val chatModelId: String? = null,
        @JsonProperty("embedding_model_id") val embeddingModelId: String? = null,
        val k: Int? = null,
        @JsonProperty("max_context_tokens") val maxContextTokens: Int? = null,
    )

    @JsonIgnoreProperties(ignoreUnknown = true)
    private data class RawLocalSearch(
        val prompt: String? = null,
        @JsonProperty("chat_model_id") val chatModelId: String? = null,
        @JsonProperty("embedding_model_id") val embeddingModelId: String? = null,
        @JsonProperty("text_unit_prop") val textUnitProp: Double? = null,
        @JsonProperty("community_prop") val communityProp: Double? = null,
        @JsonProperty("conversation_history_max_turns") val conversationHistoryMaxTurns: Int? = null,
        @JsonProperty("top_k_entities") val topKEntities: Int? = null,
        @JsonProperty("top_k_relationships") val topKRelationships: Int? = null,
        @JsonProperty("max_context_tokens") val maxContextTokens: Int? = null,
    )

    @JsonIgnoreProperties(ignoreUnknown = true)
    private data class RawGlobalSearch(
        @JsonProperty("map_prompt") val mapPrompt: String? = null,
        @JsonProperty("reduce_prompt") val reducePrompt: String? = null,
        @JsonProperty("knowledge_prompt") val knowledgePrompt: String? = null,
        @JsonProperty("chat_model_id") val chatModelId: String? = null,
        @JsonProperty("embedding_model_id") val embeddingModelId: String? = null,
        @JsonProperty("map_max_length") val mapMaxLength: Int? = null,
        @JsonProperty("reduce_max_length") val reduceMaxLength: Int? = null,
        @JsonProperty("max_context_tokens") val maxContextTokens: Int? = null,
        @JsonProperty("data_max_tokens") val dataMaxTokens: Int? = null,
        @JsonProperty("dynamic_search_threshold") val dynamicSearchThreshold: Int? = null,
        @JsonProperty("dynamic_search_keep_parent") val dynamicSearchKeepParent: Boolean? = null,
        @JsonProperty("dynamic_search_num_repeats") val dynamicSearchNumRepeats: Int? = null,
        @JsonProperty("dynamic_search_use_summary") val dynamicSearchUseSummary: Boolean? = null,
        @JsonProperty("dynamic_search_max_level") val dynamicSearchMaxLevel: Int? = null,
    )

    @JsonIgnoreProperties(ignoreUnknown = true)
    private data class RawDriftSearch(
        val prompt: String? = null,
        @JsonProperty("reduce_prompt") val reducePrompt: String? = null,
        @JsonProperty("chat_model_id") val chatModelId: String? = null,
        @JsonProperty("embedding_model_id") val embeddingModelId: String? = null,
        @JsonProperty("n_depth") val nDepth: Int? = null,
    )
}
