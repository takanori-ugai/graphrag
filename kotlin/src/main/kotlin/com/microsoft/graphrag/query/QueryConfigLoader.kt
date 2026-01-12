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
        val overrideDirs = overrideOutputDirs.map { it.toAbsolutePath().normalize() }

        if (!Files.exists(settingsPath)) {
            val indexes = buildIndexConfigs(resolvedRoot, null, overrideDirs)
            return defaultConfig(resolvedRoot, indexes)
        }

        val raw =
            runCatching {
                mapper.readValue(settingsPath.toFile(), RawConfig::class.java)
            }.getOrElse { error ->
                println("Warning: failed to parse $settingsPath ($error); using default outputs.")
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

    private fun resolveEmbedding(
        raw: RawConfig,
        id: String?,
    ): String? {
        val targetId = id ?: DEFAULT_EMBEDDING_ID
        return raw.models[targetId]?.model ?: raw.models[DEFAULT_EMBEDDING_ID]?.model
    }

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
            println("Warning: failed to load prompt at $resolved ($error); using default.")
            fallback
        }
    }

    private fun sanitizeName(name: String): String = name.trim().ifBlank { "index" }.replace("\\s+".toRegex(), "_")

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
