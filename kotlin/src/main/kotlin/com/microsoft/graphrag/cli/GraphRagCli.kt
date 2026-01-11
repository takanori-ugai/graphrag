package com.microsoft.graphrag.cli

import com.microsoft.graphrag.index.defaultEmbeddingModel
import com.microsoft.graphrag.query.BasicQueryEngine
import com.microsoft.graphrag.query.QueryIndexLoader
import dev.langchain4j.model.openai.OpenAiChatModel
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
        description = ["Index output directory (contains the parquet files)."],
    )
    var data: Path? = null

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

    override fun run() {
        val selectedMethod = method.lowercase()
        if (selectedMethod != "basic") {
            println("Only 'basic' query is implemented in Kotlin right now (requested=$method).")
            return
        }

        val outputDir = (data ?: root.resolve("sample-index/output")).toAbsolutePath().normalize()
        val indexData = QueryIndexLoader(outputDir).load()

        val apiKey =
            System.getenv("OPENAI_API_KEY")
                ?: error("OPENAI_API_KEY environment variable is required for querying.")
        val chatModel =
            OpenAiChatModel
                .builder()
                .apiKey(apiKey)
                .modelName("gpt-4o-mini")
                .build()
        val embeddingModel = defaultEmbeddingModel(apiKey)

        val engine =
            BasicQueryEngine(
                chatModel = chatModel,
                embeddingModel = embeddingModel,
                vectorStore = indexData.vectorStore,
                textUnits = indexData.textUnits,
                textEmbeddings = indexData.textEmbeddings,
                topK = 5,
            )

        val result = runBlocking { engine.answer(query, responseType) }
        println(result.answer)
        if (verbose) {
            println("\nContext chunks used:")
            result.context.forEach { chunk ->
                println("- [${chunk.id}] score=${"%.3f".format(chunk.score)} ${chunk.text.take(120)}")
            }
        }
    }
}

@Suppress("SpreadOperator")
fun main(args: Array<String>) {
    val exitCode = CommandLine(GraphRagCli()).execute(*args)
    exitProcess(exitCode)
}
