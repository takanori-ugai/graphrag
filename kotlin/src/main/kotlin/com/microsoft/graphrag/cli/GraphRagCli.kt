package com.microsoft.graphrag.cli

import com.microsoft.graphrag.index.CommunityReport
import com.microsoft.graphrag.index.defaultEmbeddingModel
import com.microsoft.graphrag.query.BasicQueryEngine
import com.microsoft.graphrag.query.CollectingQueryCallbacks
import com.microsoft.graphrag.query.DriftSearchEngine
import com.microsoft.graphrag.query.GlobalSearchEngine
import com.microsoft.graphrag.query.LocalQueryEngine
import com.microsoft.graphrag.query.ModelParams
import com.microsoft.graphrag.query.QueryCallbacks
import com.microsoft.graphrag.query.QueryConfigLoader
import com.microsoft.graphrag.query.QueryIndexLoader
import com.microsoft.graphrag.query.QueryResult
import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.model.openai.OpenAiStreamingChatModel
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.runBlocking
import picocli.CommandLine
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import java.nio.file.Path
import kotlin.system.exitProcess

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
        description = ["Desired response format (e.g. 'Single Sentence')."],
        defaultValue = "Multiple Paragraphs",
    )
    var responseType: String = "Multiple Paragraphs"

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

    override fun run() {
        val selectedMethod = method.lowercase()

        val overrideOutputs = data.map { it.toAbsolutePath().normalize() }
        val queryConfig = QueryConfigLoader.load(root, config, overrideOutputs)
        val indexData = QueryIndexLoader(queryConfig.outputDirs).load()

        val apiKey = System.getenv("OPENAI_API_KEY") ?: error("OPENAI_API_KEY environment variable is required for querying.")
        val chatModelName = queryConfig.chatModel ?: "gpt-4o-mini"
        val embeddingModelName = queryConfig.embeddingModel ?: "text-embedding-3-small"
        val chatModel =
            OpenAiChatModel
                .builder()
                .apiKey(apiKey)
                .modelName(chatModelName)
                .build()
        val streamingChatModel =
            OpenAiStreamingChatModel
                .builder()
                .apiKey(apiKey)
                .modelName(chatModelName)
                .build()
        val embeddingModel = defaultEmbeddingModel(apiKey, embeddingModelName)
        val callbacks = CollectingQueryCallbacks()
        val callbackList: List<QueryCallbacks> = listOf(callbacks)
        val filteredReports = filterCommunityReports(indexData.communityReports, indexData.communityHierarchy, communityLevel)

        val result: QueryResult =
            when (selectedMethod) {
                "basic" -> {
                    runBlocking {
                        val engine =
                            BasicQueryEngine(
                                streamingModel = streamingChatModel,
                                embeddingModel = embeddingModel,
                                vectorStore = indexData.vectorStore,
                                textUnits = indexData.textUnits,
                                textEmbeddings = indexData.textEmbeddings,
                                topK = 5,
                                callbacks = callbackList,
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
                        val engine =
                            LocalQueryEngine(
                                streamingModel = streamingChatModel,
                                embeddingModel = embeddingModel,
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
                                modelParams = ModelParams(jsonResponse = false),
                                callbacks = callbackList,
                            )
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
                        val engine =
                            GlobalSearchEngine(
                                streamingModel = streamingChatModel,
                                communityReports = filteredReports,
                                communityReportEmbeddings = indexData.communityReportEmbeddings,
                                embeddingModel = embeddingModel,
                                communityHierarchy = indexData.communityHierarchy,
                                communityLevel = communityLevel,
                                dynamicCommunitySelection = dynamicCommunitySelection,
                                callbacks = callbackList,
                                responseType = responseType,
                            )
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
                        val localEngine =
                            LocalQueryEngine(
                                streamingModel = streamingChatModel,
                                embeddingModel = embeddingModel,
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
                                modelParams = ModelParams(jsonResponse = false),
                                callbacks = driftCallbackList,
                            )
                        if (streaming) {
                            val engine =
                                DriftSearchEngine(
                                    streamingModel = streamingChatModel,
                                    communityReports = filteredReports,
                                    globalSearchEngine =
                                        GlobalSearchEngine(
                                            streamingModel = streamingChatModel,
                                            communityReports = filteredReports,
                                            communityReportEmbeddings = indexData.communityReportEmbeddings,
                                            embeddingModel = embeddingModel,
                                            communityHierarchy = indexData.communityHierarchy,
                                            communityLevel = communityLevel,
                                            dynamicCommunitySelection = dynamicCommunitySelection,
                                            callbacks = driftCallbackList,
                                            responseType = responseType,
                                        ),
                                    localQueryEngine = localEngine,
                                    callbacks = driftCallbackList,
                                )
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
                            val engine =
                                DriftSearchEngine(
                                    streamingModel = streamingChatModel,
                                    communityReports = filteredReports,
                                    globalSearchEngine =
                                        GlobalSearchEngine(
                                            streamingModel = streamingChatModel,
                                            communityReports = filteredReports,
                                            communityReportEmbeddings = indexData.communityReportEmbeddings,
                                            embeddingModel = embeddingModel,
                                            communityHierarchy = indexData.communityHierarchy,
                                            communityLevel = communityLevel,
                                            dynamicCommunitySelection = dynamicCommunitySelection,
                                            callbacks = driftCallbackList,
                                            responseType = responseType,
                                        ),
                                    localQueryEngine = localEngine,
                                    callbacks = driftCallbackList,
                                )
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
                    println("Unsupported query method: $method. Use one of: basic, local, global, drift.")
                    return
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
        if (streaming) {
            println()
        } else {
            println(finalResult.answer)
        }
        if (verbose && finalResult.context.isNotEmpty()) {
            println("\nContext chunks used:")
            finalResult.context.forEach { chunk ->
                println("- [${chunk.id}] score=${"%.3f".format(chunk.score)} ${chunk.text.take(120)}")
            }
        }
        if (verbose && finalResult.contextRecords.isNotEmpty()) {
            println("\nContext records:")
            finalResult.contextRecords.forEach { (name, records) ->
                val inContext = records.count { it["in_context"] == "true" }
                println("- $name: $inContext/${records.size} rows marked in_context")
            }
        }
    }

    private fun filterCommunityReports(
        reports: List<CommunityReport>,
        hierarchy: Map<Int, Int>,
        level: Int?,
    ): List<CommunityReport> {
        if (level == null || level < 0) return reports
        if (hierarchy.isEmpty()) return reports
        val cache = mutableMapOf<Int, Int>()

        fun depth(id: Int): Int =
            cache.getOrPut(id) {
                val parent = hierarchy[id] ?: return@getOrPut 0
                if (parent < 0) 0 else 1 + depth(parent)
            }

        return reports.filter { depth(it.communityId) == level }
    }
}

@Suppress("SpreadOperator")
fun main(args: Array<String>) {
    val exitCode = CommandLine(GraphRagCli()).execute(*args)
    exitProcess(exitCode)
}
