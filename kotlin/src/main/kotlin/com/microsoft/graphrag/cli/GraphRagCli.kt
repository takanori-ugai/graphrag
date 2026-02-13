package com.microsoft.graphrag.cli

import com.microsoft.graphrag.index.CommunityReport
import com.microsoft.graphrag.index.defaultEmbeddingModel
import com.microsoft.graphrag.query.BasicQueryEngine
import com.microsoft.graphrag.query.CollectingQueryCallbacks
import com.microsoft.graphrag.query.DriftSearchEngine
import com.microsoft.graphrag.query.GlobalSearchEngine
import com.microsoft.graphrag.query.IndexLookup
import com.microsoft.graphrag.query.LocalQueryEngine
import com.microsoft.graphrag.query.ModelParams
import com.microsoft.graphrag.query.QueryCallbacks
import com.microsoft.graphrag.query.QueryConfigLoader
import com.microsoft.graphrag.query.QueryIndexLoader
import com.microsoft.graphrag.query.QueryModelConfig
import com.microsoft.graphrag.query.QueryResult
import dev.langchain4j.model.openai.OpenAiStreamingChatModel
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.runBlocking
import picocli.CommandLine
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import java.nio.file.Path
import kotlin.system.exitProcess

/**
 * Root command for the GraphRAG CLI.
 */
@Command(
    name = "graphrag",
    description = ["GraphRAG: A graph-based retrieval-augmented generation (RAG) system."],
    mixinStandardHelpOptions = true,
    subcommands = [
        InitCommand::class,
        IndexCommand::class,
        UpdateCommand::class,
        PromptTuneCommand::class,
        QueryCommand::class,
    ],
)
class GraphRagCli : Runnable {
    override fun run() {
        // Root command only shows help by default.
    }
}

/**
 * CLI command that generates a default configuration template.
 *
 * @property root Project root directory where configuration will be generated.
 * @property force Whether to overwrite existing configuration.
 */
@Command(
    name = "init",
    description = ["Generate a default configuration file."],
    mixinStandardHelpOptions = true,
)
class InitCommand : Runnable {
    @Option(
        names = ["-r", "--root"],
        description = ["Project root directory."],
        defaultValue = ".",
    )
    var root: Path = Path.of(".")

    @Option(
        names = ["-f", "--force"],
        description = ["Force initialization even if the project already exists."],
    )
    var force: Boolean = false

    override fun run() {
        println("init not yet implemented (root=$root, force=$force)")
    }
}

/**
 * Shared CLI options for indexing-related commands.
 *
 * @property config Optional configuration file path.
 * @property root Project root directory.
 * @property verbose Whether to enable verbose logging.
 * @property memprofile Whether to enable memory profiling.
 * @property cache Whether LLM cache usage is enabled.
 * @property skipValidation Whether to skip preflight validation.
 * @property output Optional output directory override.
 */
class SharedIndexOptions {
    @Option(
        names = ["-c", "--config"],
        description = ["Configuration file to use."],
    )
    var config: Path? = null

    @Option(
        names = ["-r", "--root"],
        description = ["Project root directory."],
        defaultValue = ".",
    )
    var root: Path = Path.of(".")

    @Option(
        names = ["-v", "--verbose"],
        description = ["Run with verbose logging."],
    )
    var verbose: Boolean = false

    @Option(
        names = ["--memprofile"],
        description = ["Run the indexing pipeline with memory profiling."],
    )
    var memprofile: Boolean = false

    @Option(
        names = ["--cache"],
        description = ["Use LLM cache (disable with --no-cache)."],
        negatable = true,
        defaultValue = "true",
    )
    var cache: Boolean = true

    @Option(
        names = ["--skip-validation"],
        description = ["Skip preflight validation (useful when running no LLM steps)."],
    )
    var skipValidation: Boolean = false

    @Option(
        names = ["-o", "--output"],
        description = ["Indexing pipeline output directory (overrides output.base_dir)."],
    )
    var output: Path? = null
}

/**
 * CLI command that builds a new index.
 *
 * @property method Indexing method identifier.
 * @property dryRun Whether to validate configuration without executing steps.
 */
@Command(
    name = "index",
    description = ["Build a knowledge graph index."],
    mixinStandardHelpOptions = true,
)
class IndexCommand : Runnable {
    @CommandLine.Mixin
    private val shared = SharedIndexOptions()

    @Option(
        names = ["-m", "--method"],
        description = ["Indexing method to use (standard, fast, standard-update, fast-update)."],
        defaultValue = "standard",
    )
    var method: String = "standard"

    @Option(
        names = ["--dry-run"],
        description = ["Validate configuration without executing any steps."],
    )
    var dryRun: Boolean = false

    override fun run() {
        IndexRunner().run(
            IndexOptions(
                root = shared.root,
                config = shared.config,
                method = method,
                verbose = shared.verbose,
                memprofile = shared.memprofile,
                dryRun = dryRun,
                cache = shared.cache,
                skipValidation = shared.skipValidation,
                output = shared.output,
                isUpdate = false,
            ),
        )
    }
}

/**
 * CLI command that updates an existing index.
 *
 * @property method Indexing method identifier.
 */
@Command(
    name = "update",
    description = ["Update an existing knowledge graph index."],
    mixinStandardHelpOptions = true,
)
class UpdateCommand : Runnable {
    @CommandLine.Mixin
    private val shared = SharedIndexOptions()

    @Option(
        names = ["-m", "--method"],
        description = ["Indexing method to use (standard-update, fast-update, etc.)."],
        defaultValue = "standard-update",
    )
    var method: String = "standard-update"

    override fun run() {
        IndexRunner().run(
            IndexOptions(
                root = shared.root,
                config = shared.config,
                method = method,
                verbose = shared.verbose,
                memprofile = shared.memprofile,
                dryRun = false,
                cache = shared.cache,
                skipValidation = shared.skipValidation,
                output = shared.output,
                isUpdate = true,
            ),
        )
    }
}

/**
 * CLI command that generates tuned prompt templates from project data.
 *
 * @property root Project root directory.
 * @property config Optional configuration file path.
 * @property verbose Whether to enable verbose logging.
 * @property domain Domain label associated with the input data.
 * @property selectionMethod Text chunk selection strategy.
 * @property nSubsetMax Max chunks to embed when using auto selection.
 * @property k Max docs to select per centroid when using auto selection.
 * @property limit Max documents to load for random/top selection.
 * @property maxTokens Maximum token budget for prompt generation.
 * @property minExamplesRequired Minimum number of examples to generate.
 * @property chunkSize Size of example text chunks.
 * @property overlap Overlap between consecutive chunks.
 * @property language Primary language for inputs and outputs.
 * @property discoverEntityTypes Whether to infer additional entity types.
 * @property output Output directory (relative to project root) for prompts.
 */
@Command(
    name = "prompt-tune",
    description = ["Generate custom GraphRAG prompts with your own data (auto templating)."],
    mixinStandardHelpOptions = true,
)
class PromptTuneCommand : Runnable {
    @Option(
        names = ["-r", "--root"],
        description = ["Project root directory."],
        defaultValue = ".",
    )
    var root: Path = Path.of(".")

    @Option(
        names = ["-c", "--config"],
        description = ["Configuration file to use."],
    )
    var config: Path? = null

    @Option(
        names = ["-v", "--verbose"],
        description = ["Run with verbose logging."],
    )
    var verbose: Boolean = false

    @Option(
        names = ["--domain"],
        description = ["Domain your input data is related to."],
    )
    var domain: String? = null

    @Option(
        names = ["--selection-method"],
        description = ["Text chunk selection method."],
        defaultValue = "random",
    )
    var selectionMethod: String = "random"

    @Option(
        names = ["--n-subset-max"],
        description = ["Number of text chunks to embed when selection-method=auto."],
        defaultValue = "100",
    )
    var nSubsetMax: Int = 100

    @Option(
        names = ["--k"],
        description = ["Max docs to select from each centroid when selection-method=auto."],
        defaultValue = "5",
    )
    var k: Int = 5

    @Option(
        names = ["--limit"],
        description = ["Number of documents to load when selection-method=random/top."],
        defaultValue = "100",
    )
    var limit: Int = 100

    @Option(
        names = ["--max-tokens"],
        description = ["Max token count for prompt generation."],
        defaultValue = "4096",
    )
    var maxTokens: Int = 4096

    @Option(
        names = ["--min-examples-required"],
        description = ["Minimum number of examples to generate/include in the prompt."],
        defaultValue = "2",
    )
    var minExamplesRequired: Int = 2

    @Option(
        names = ["--chunk-size"],
        description = ["Chunk size for example text chunks."],
        defaultValue = "1024",
    )
    var chunkSize: Int = 1024

    @Option(
        names = ["--overlap"],
        description = ["Chunk overlap for example text chunks."],
        defaultValue = "128",
    )
    var overlap: Int = 128

    @Option(
        names = ["--language"],
        description = ["Primary language for inputs and outputs in prompts."],
    )
    var language: String? = null

    @Option(
        names = ["--discover-entity-types"],
        description = ["Discover and extract unspecified entity types (disable with --no-discover-entity-types)."],
        negatable = true,
        defaultValue = "true",
    )
    var discoverEntityTypes: Boolean = true

    @Option(
        names = ["-o", "--output"],
        description = ["Directory to save prompts to, relative to the project root."],
        defaultValue = "prompts",
    )
    var output: Path = Path.of("prompts")

    override fun run() {
        println(
            "prompt-tune not yet implemented (root=$root, config=$config, selection=$selectionMethod, " +
                "limit=$limit, k=$k, nSubsetMax=$nSubsetMax, maxTokens=$maxTokens, chunkSize=$chunkSize, " +
                "overlap=$overlap, language=$language, discoverEntityTypes=$discoverEntityTypes, " +
                "output=$output, verbose=$verbose, domain=$domain, minExamplesRequired=$minExamplesRequired)",
        )
    }
}

/**
 * CLI command that executes a query against an index.
 *
 * @property method Query algorithm to use (basic, local, global, drift).
 * @property query Query text to execute.
 * @property config Optional configuration file path.
 * @property verbose Whether to enable verbose logging.
 * @property data Index output directories that contain parquet files.
 * @property root Project root directory.
 * @property communityLevel Community hierarchy level used for reports.
 * @property dynamicCommunitySelection Whether to enable dynamic community selection.
 * @property responseType Desired response format for the LLM.
 * @property streaming Whether to stream the answer token-by-token.
 * @property driftQuery Optional override query for drift/local searches.
 */
@Command(
    name = "query",
    description = ["Query a knowledge graph index."],
    mixinStandardHelpOptions = true,
)
class QueryCommand : Runnable {
    @Option(
        names = ["-m", "--method"],
        description = ["Query algorithm to use (local, global, drift, basic)."],
        required = true,
    )
    lateinit var method: String

    @Option(
        names = ["-q", "--query"],
        description = ["Query to execute."],
        required = true,
    )
    lateinit var query: String

    @Option(
        names = ["-c", "--config"],
        description = ["Configuration file to use."],
    )
    var config: Path? = null

    @Option(
        names = ["-v", "--verbose"],
        description = ["Run with verbose logging."],
    )
    var verbose: Boolean = false

    @Option(
        names = ["-d", "--data"],
        description = ["Index output directories (comma-separated) that contain the parquet files."],
        split = ",",
    )
    var data: List<Path> = emptyList()

    @Option(
        names = ["-r", "--root"],
        description = ["Project root directory."],
        defaultValue = ".",
    )
    var root: Path = Path.of(".")

    @Option(
        names = ["--community-level"],
        description = ["Leiden hierarchy level from which to load community reports."],
        defaultValue = "2",
    )
    var communityLevel: Int = 2

    @Option(
        names = ["--dynamic-community-selection"],
        description = ["Use global search with dynamic community selection (disable with --no-dynamic-community-selection)."],
        negatable = true,
        defaultValue = "false",
    )
    var dynamicCommunitySelection: Boolean = false

    @Option(
        names = ["--response-type"],
        description = ["Desired response format (e.g. 'JSON response (response, score, follow_up_queries)')."],
        defaultValue = "JSON response (response, score, follow_up_queries)",
    )
    var responseType: String = "JSON response (response, score, follow_up_queries)"

    @Option(
        names = ["--streaming"],
        description = ["Print the response in a streaming manner (disable with --no-streaming)."],
        negatable = true,
        defaultValue = "false",
    )
    var streaming: Boolean = false

    @Option(
        names = ["--drift-query"],
        description = ["Override the global query used when running drift/local searches."],
    )
    var driftQuery: String? = null

    /**
     * Execute the CLI "query" command: load configuration and index data, build models and engines,
     * run the selected query method (basic, local, global, or drift) with optional streaming, and print the result.
     *
     * The command loads query and index configuration, constructs embedding and streaming chat models,
     * selects and runs the appropriate query engine, enriches context records with index names, and outputs
     * the final answer and (when verbose) context details.
     *
     * @throws IllegalStateException if the OPENAI_API_KEY environment variable is not set.
     * @throws CommandLine.ParameterException if the selected query method is unsupported.
     */
    @Suppress("LongMethod", "CyclomaticComplexMethod")
    override fun run() {
        val selectedMethod = method.lowercase()

        val overrideOutputs = data.map { it.toAbsolutePath().normalize() }
        val queryConfig = QueryConfigLoader.load(root, config, overrideOutputs)
        val indexData = QueryIndexLoader(queryConfig.indexes).load()

        val apiKey = System.getenv("OPENAI_API_KEY") ?: error("OPENAI_API_KEY environment variable is required for querying.")
        val defaultChat = queryConfig.defaultChatModel ?: QueryModelConfig("gpt-4o-mini")
        val defaultEmbeddingName = queryConfig.defaultEmbeddingModel ?: "text-embedding-3-small"

        fun buildStreamingModel(modelConfig: QueryModelConfig?): OpenAiStreamingChatModel {
            val name = modelConfig?.model ?: defaultChat.model ?: "gpt-4o-mini"
            val params = modelConfig?.params ?: defaultChat.params
            val builder =
                OpenAiStreamingChatModel
                    .builder()
                    .apiKey(apiKey)
                    .modelName(name)
            params.temperature?.let { builder.temperature(it) }
            params.topP?.let { builder.topP(it) }
            params.maxTokens?.let { builder.maxTokens(it) }
            return builder.build()
        }

        fun buildEmbeddingModel(name: String?): dev.langchain4j.model.embedding.EmbeddingModel =
            defaultEmbeddingModel(apiKey, name ?: defaultEmbeddingName)

        val callbacks = CollectingQueryCallbacks()
        val callbackList: List<QueryCallbacks> = listOf(callbacks)
        val filteredReports = filterCommunityReports(indexData.communityReports, indexData.communityHierarchy, communityLevel)

        fun createLocalEngine(callbacks: List<QueryCallbacks>): LocalQueryEngine {
            val modelConfig = queryConfig.local.chat ?: defaultChat
            val localParams = modelConfig.params
            return LocalQueryEngine(
                streamingModel = buildStreamingModel(modelConfig),
                embeddingModel = buildEmbeddingModel(queryConfig.local.embeddingModel),
                vectorStore = indexData.vectorStore,
                textUnits = indexData.textUnits,
                textEmbeddings = indexData.textEmbeddings,
                entities = indexData.entities,
                entitySummaries = indexData.entitySummaries,
                relationships = indexData.relationships,
                claims = indexData.claims,
                covariates = indexData.covariates,
                communities = indexData.communities,
                communityReports = filteredReports,
                modelParams = localParams.copy(jsonResponse = localParams.jsonResponse),
                topKEntities = queryConfig.local.topKEntities,
                topKRelationships = queryConfig.local.topKRelationships,
                maxContextTokens = queryConfig.local.maxContextTokens,
                systemPrompt = queryConfig.local.prompt,
                textUnitProp = queryConfig.local.textUnitProp,
                communityProp = queryConfig.local.communityProp,
                conversationHistoryMaxTurns = queryConfig.local.conversationHistoryMaxTurns,
                callbacks = callbacks,
            )
        }

        fun createGlobalEngine(
            callbacks: List<QueryCallbacks>,
            mapParamsOverride: ModelParams? = null,
            reduceParamsOverride: ModelParams? = null,
        ): GlobalSearchEngine {
            val modelConfig = queryConfig.global.chat ?: defaultChat
            val baseParams = modelConfig.params
            val mapParams = mapParamsOverride ?: baseParams.copy(jsonResponse = true)
            val reduceParams = reduceParamsOverride ?: baseParams.copy(jsonResponse = false)
            return GlobalSearchEngine(
                streamingModel = buildStreamingModel(modelConfig),
                communityReports = filteredReports,
                communityHierarchy = indexData.communityHierarchy,
                communityLevel = communityLevel,
                dynamicCommunitySelection = dynamicCommunitySelection,
                dynamicThreshold = queryConfig.global.dynamic.threshold,
                dynamicKeepParent = queryConfig.global.dynamic.keepParent,
                dynamicNumRepeats = queryConfig.global.dynamic.numRepeats,
                dynamicUseSummary = queryConfig.global.dynamic.useSummary,
                dynamicMaxLevel = queryConfig.global.dynamic.maxLevel,
                callbacks = callbacks,
                responseType = responseType,
                allowGeneralKnowledge = queryConfig.global.allowGeneralKnowledge,
                generalKnowledgeInstruction =
                    queryConfig.global.knowledgePrompt
                        ?: GlobalSearchEngine.DEFAULT_GENERAL_KNOWLEDGE_INSTRUCTION,
                mapSystemPrompt = queryConfig.global.mapPrompt,
                reduceSystemPrompt = queryConfig.global.reducePrompt,
                mapMaxLength = queryConfig.global.mapMaxLength,
                reduceMaxLength = queryConfig.global.reduceMaxLength,
                maxContextTokens = queryConfig.global.maxContextTokens,
                maxDataTokens = queryConfig.global.dataMaxTokens,
                mapParams = mapParams,
                reduceParams = reduceParams,
            )
        }

        val result: QueryResult =
            when (selectedMethod) {
                "basic" -> {
                    runBlocking {
                        val modelConfig = queryConfig.basic.chat ?: defaultChat
                        val engine =
                            BasicQueryEngine(
                                streamingModel = buildStreamingModel(modelConfig),
                                embeddingModel = buildEmbeddingModel(queryConfig.basic.embeddingModel),
                                vectorStore = indexData.vectorStore,
                                textUnits = indexData.textUnits,
                                textEmbeddings = indexData.textEmbeddings,
                                topK = queryConfig.basic.k,
                                maxContextTokens = queryConfig.basic.maxContextTokens,
                                callbacks = callbackList,
                                systemPrompt = queryConfig.basic.prompt,
                            )
                        if (streaming) {
                            val builder = StringBuilder()
                            engine.streamAnswer(query, responseType).collect { partial ->
                                print(partial)
                                builder.append(partial)
                            }
                            QueryResult(
                                answer = builder.toString(),
                                context = emptyList(),
                                contextRecords = callbacks.contextRecords,
                            )
                        } else {
                            engine.answer(query, responseType)
                        }
                    }
                }

                "local" -> {
                    runBlocking {
                        val engine = createLocalEngine(callbackList)
                        if (streaming) {
                            val builder = StringBuilder()
                            engine.streamAnswer(query, responseType, driftQuery = driftQuery).collect { partial ->
                                print(partial)
                                builder.append(partial)
                            }
                            QueryResult(
                                answer = builder.toString(),
                                context = emptyList(),
                                contextRecords = callbacks.contextRecords,
                            )
                        } else {
                            engine.answer(query, responseType, driftQuery = driftQuery)
                        }
                    }
                }

                "global" -> {
                    runBlocking {
                        val engine = createGlobalEngine(callbackList)
                        if (streaming) {
                            val builder = StringBuilder()
                            engine.streamSearch(query).collect { partial ->
                                print(partial)
                                builder.append(partial)
                            }
                            QueryResult(
                                answer = builder.toString(),
                                context = emptyList(),
                                contextRecords = callbacks.contextRecords,
                                contextText = callbacks.reduceContext,
                            )
                        } else {
                            val globalResult = engine.search(query)
                            QueryResult(
                                answer = globalResult.answer,
                                context = emptyList(),
                                contextRecords = globalResult.contextRecords,
                                contextText = globalResult.reduceContextText,
                                llmCalls = globalResult.llmCalls,
                                promptTokens = globalResult.promptTokens,
                                outputTokens = globalResult.outputTokens,
                                llmCallsCategories = globalResult.llmCallsCategories,
                                promptTokensCategories = globalResult.promptTokensCategories,
                                outputTokensCategories = globalResult.outputTokensCategories,
                            )
                        }
                    }
                }

                "drift" -> {
                    runBlocking {
                        val driftCallbacks = CollectingQueryCallbacks()
                        val driftCallbackList: List<QueryCallbacks> = listOf(driftCallbacks)
                        val localEngine = createLocalEngine(driftCallbackList)
                        val globalEngine = createGlobalEngine(driftCallbackList)
                        val driftModel = queryConfig.drift.chat ?: defaultChat
                        val engine =
                            DriftSearchEngine(
                                streamingModel = buildStreamingModel(driftModel),
                                communityReports = filteredReports,
                                globalSearchEngine = globalEngine,
                                localQueryEngine = localEngine,
                                primerSystemPrompt = queryConfig.drift.prompt,
                                reduceSystemPrompt = queryConfig.drift.reducePrompt,
                                callbacks = driftCallbackList,
                                maxIterations = queryConfig.drift.maxIterations,
                            )
                        if (streaming) {
                            val builder = StringBuilder()
                            engine.streamSearch(query, followUpQueries = driftQuery?.let { listOf(it) } ?: emptyList()).collect { partial ->
                                print(partial)
                                builder.append(partial)
                            }
                            QueryResult(
                                answer = builder.toString(),
                                context = emptyList(),
                                contextRecords = driftCallbacks.contextRecords,
                                contextText = driftCallbacks.reduceContext,
                            )
                        } else {
                            val driftResult =
                                engine.search(
                                    question = query,
                                    followUpQueries = driftQuery?.let { listOf(it) } ?: emptyList(),
                                )
                            QueryResult(
                                answer = driftResult.answer,
                                context = emptyList(),
                                contextRecords = driftCallbacks.contextRecords,
                                contextText = driftCallbacks.reduceContext,
                                llmCalls = driftResult.llmCalls,
                                promptTokens = driftResult.promptTokens,
                                outputTokens = driftResult.outputTokens,
                                llmCallsCategories = driftResult.llmCallsCategories,
                                promptTokensCategories = driftResult.promptTokensCategories,
                                outputTokensCategories = driftResult.outputTokensCategories,
                            )
                        }
                    }
                }

                else -> {
                    System.err.println("Unsupported query method: $method. Use one of: basic, local, global, drift.")
                    throw CommandLine.ParameterException(CommandLine(this), "Unsupported query method: $method")
                }
            }
        val finalResult =
            if (result.contextRecords.isEmpty() && callbacks.contextRecords.isNotEmpty()) {
                result.copy(
                    contextRecords = callbacks.contextRecords,
                    contextText = if (result.contextText.isNotBlank()) result.contextText else callbacks.reduceContext,
                )
            } else {
                result
            }
        val enrichedContext = attachIndexNames(finalResult.contextRecords, indexData.indexLookup)
        val mergedResult = finalResult.copy(contextRecords = enrichedContext)
        if (streaming) {
            println()
        } else {
            println(mergedResult.answer)
        }
        if (mergedResult.score != null || mergedResult.followUpQueries.isNotEmpty()) {
            println()
            if (mergedResult.score != null) {
                println("Score: ${"%.2f".format(mergedResult.score)}")
            }
            if (mergedResult.followUpQueries.isNotEmpty()) {
                println("Follow-up queries:")
                mergedResult.followUpQueries.forEachIndexed { idx, q ->
                    println("${idx + 1}. $q")
                }
            }
        }
        if (verbose && mergedResult.context.isNotEmpty()) {
            println("\nContext chunks used:")
            mergedResult.context.forEach { chunk ->
                println("- [${chunk.id}] score=${"%.3f".format(chunk.score)} ${chunk.text.take(120)}")
            }
        }
        if (verbose && mergedResult.contextRecords.isNotEmpty()) {
            println("\nContext records:")
            mergedResult.contextRecords.forEach { (name, records) ->
                val inContext = records.count { it["in_context"] == "true" }
                println("- $name: $inContext/${records.size} rows marked in_context")
            }
        }
    }

    /**
     * Annotates context records with their originating index name when it can be determined.
     *
     * For each record in `contextRecords`, attempts to locate an index name in `lookup` using
     * common identifier fields (for example `id`, `entity`, `entity_id`, `chunk_id`,
     * `community`, `community_id`) and, when found, returns a copy of the record with an
     * added `"index_name"` entry. If `contextRecords` is empty or `lookup` contains one or
     * zero index names, the input is returned unchanged.
     *
     * @param contextRecords A map from source name to a list of context records (each record is a map of string
     * keys to values).
     * @param lookup An IndexLookup providing mappings from identifiers to index names used to resolve records' origins.
     * @return A map with the same structure as `contextRecords` where records that could be resolved include an
     * `"index_name"` key; unresolved records are unchanged.
     */
    @Suppress("NestedBlockDepth")
    private fun attachIndexNames(
        contextRecords: Map<String, List<Map<String, String>>>,
        lookup: IndexLookup,
    ): Map<String, List<Map<String, String>>> {
        if (contextRecords.isEmpty() || lookup.indexNames.size <= 1) return contextRecords

        @Suppress("ReturnCount")
        fun findIndex(record: Map<String, String>): String? {
            val candidates =
                listOf(
                    record["id"],
                    record["entity"],
                    record["entity_id"],
                    record["chunk_id"],
                    record["community"],
                    record["community_id"],
                )
            candidates.forEach { value ->
                if (!value.isNullOrBlank()) {
                    lookup.reportIndex[value]?.let { return it }
                    lookup.entityIndex[value]?.let { return it }
                    lookup.textUnitIndex[value]?.let { return it }
                    value.toIntOrNull()?.let { intId ->
                        lookup.communityIndex[intId]?.let { return it }
                    }
                }
            }
            return null
        }

        return contextRecords.mapValues { (_, records) ->
            records.map { record ->
                val indexName = findIndex(record) ?: return@map record
                record.toMutableMap().apply { this["index_name"] = indexName }.toMap()
            }
        }
    }

    /**
     * Filter community reports to those whose community is at the specified depth in the hierarchy.
     *
     * @param reports The list of community reports to filter.
     * @param hierarchy A map from community id to its parent community id; a negative parent or missing entry
     * terminates traversal.
     * @param level The target depth (0 = root). If `null` or negative, or if `hierarchy` is empty, the original
     * `reports` list is returned unchanged.
     * @return A list of `CommunityReport` instances whose community depth equals `level`.
     */
    @Suppress("ReturnCount", "LoopWithTooManyJumpStatements")
    private fun filterCommunityReports(
        reports: List<CommunityReport>,
        hierarchy: Map<Int, Int>,
        level: Int?,
    ): List<CommunityReport> {
        if (level == null || level < 0) return reports
        if (hierarchy.isEmpty()) return reports
        val cache = mutableMapOf<Int, Int>()

        fun depth(id: Int): Int {
            cache[id]?.let { return it }
            var current = id
            var d = 0
            val seen = mutableSetOf<Int>()
            while (true) {
                if (!seen.add(current)) break
                val parent = hierarchy[current] ?: break
                if (parent < 0) break
                current = parent
                d++
            }
            cache[id] = d
            return d
        }

        return reports.filter { depth(it.communityId) == level }
    }
}

/**
 * CLI entrypoint that delegates to Picocli.
 *
 * @param args Raw CLI arguments.
 */
@Suppress("SpreadOperator")
fun main(args: Array<String>) {
    val exitCode = CommandLine(GraphRagCli()).execute(*args)
    exitProcess(exitCode)
}
