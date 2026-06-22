package com.microsoft.graphrag.eval

import com.microsoft.graphrag.index.CommunityReport
import com.microsoft.graphrag.index.defaultEmbeddingModel
import com.microsoft.graphrag.query.BasicQueryEngine
import com.microsoft.graphrag.query.CollectingQueryCallbacks
import com.microsoft.graphrag.query.DriftSearchEngine
import com.microsoft.graphrag.query.GlobalSearchEngine
import com.microsoft.graphrag.query.LocalQueryEngine
import com.microsoft.graphrag.query.ModelParams
import com.microsoft.graphrag.query.QueryConfig
import com.microsoft.graphrag.query.QueryConfigLoader
import com.microsoft.graphrag.query.QueryIndexData
import com.microsoft.graphrag.query.QueryIndexLoader
import com.microsoft.graphrag.query.QueryModelConfig
import com.microsoft.graphrag.query.QueryResult
import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.model.openai.OpenAiStreamingChatModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonPrimitive
import ragas.evaluate
import ragas.llms.LangChain4jLlm
import ragas.metrics.collections.AnswerCorrectnessMetric
import ragas.metrics.collections.FactualCorrectnessMetric
import ragas.model.EvaluationDataset
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig
import java.nio.file.Files
import java.nio.file.Path
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.DoubleAdder
import kotlin.math.min

@Serializable
data class MusiqueParagraph(
    val idx: Int,
    val title: String,
    @SerialName("paragraph_text")
    val paragraphText: String,
    @SerialName("is_supporting")
    val isSupporting: Boolean,
)

@Serializable
data class MusiqueExample(
    val id: String,
    val paragraphs: List<MusiqueParagraph>,
    val question: String,
    val answer: String,
    @SerialName("answer_aliases")
    val answerAliases: List<String> = emptyList(),
    val answerable: Boolean = true,
)

private enum class EvalQueryMethod {
    BASIC,
    LOCAL,
    GLOBAL,
    DRIFT,
}

private data class CliArgs(
    val inputPath: Path,
    val root: Path,
    val configPath: Path?,
    val outputDirs: List<Path>,
)

private data class GeneratedSample(
    val id: String,
    val question: String,
    val prediction: String,
    val reference: String,
    val primaryGold: String,
    val aliases: List<String>,
    val retrievedContexts: List<String>,
    val exactMatch: Double,
)

/**
 * Runs MusiQue evaluation using GraphRAG generation and ragas scoring.
 *
 * Environment variables:
 * - OPENAI_API_KEY (required)
 * - MUSIQUE_QUERY_METHOD (basic|local|global|drift, default: local)
 * - MUSIQUE_LIMIT (optional)
 * - MUSIQUE_PARALLELISM (default: 1)
 * - MUSIQUE_TOP_K_CONTEXTS (default: 10)
 * - MUSIQUE_COMMUNITY_LEVEL (default: 2)
 * - MUSIQUE_DYNAMIC_COMMUNITY_SELECTION (true|false, default: false)
 * - MUSIQUE_RESPONSE_TYPE (default: JSON response (response, score, follow_up_queries))
 * - MUSIQUE_CHAT_MODEL (default: config default chat model or gpt-4o-mini)
 * - MUSIQUE_EMBEDDING_MODEL (default: config default embedding model or text-embedding-3-small)
 * - MUSIQUE_EVAL_MODEL (default: MUSIQUE_CHAT_MODEL)
 */
object MusiQueGraphRagRagas {
    private val inputJson = Json { ignoreUnknownKeys = true }

    @Suppress("LongMethod", "TooGenericExceptionCaught")
    @JvmStatic
    fun main(args: Array<String>) {
        val cliArgs = parseCliArgs(args)
        if (!Files.exists(cliArgs.inputPath)) {
            System.err.println("Missing MusiQue data file at ${cliArgs.inputPath}")
            kotlin.system.exitProcess(1)
        }

        val apiKey = System.getenv("OPENAI_API_KEY")
        if (apiKey.isNullOrBlank()) {
            System.err.println("OPENAI_API_KEY is required for generation and ragas evaluation")
            kotlin.system.exitProcess(1)
        }

        val method = parseMethod(System.getenv("MUSIQUE_QUERY_METHOD"))
        val limit = (System.getenv("MUSIQUE_LIMIT") ?: "").toIntOrNull()
        val parallelism = (System.getenv("MUSIQUE_PARALLELISM") ?: "1").toIntOrNull() ?: 1
        val topKContexts = (System.getenv("MUSIQUE_TOP_K_CONTEXTS") ?: "10").toIntOrNull() ?: 10
        val communityLevel = (System.getenv("MUSIQUE_COMMUNITY_LEVEL") ?: "2").toIntOrNull() ?: 2
        val dynamicCommunitySelection = (System.getenv("MUSIQUE_DYNAMIC_COMMUNITY_SELECTION") ?: "false").toBoolean()
        val responseType =
            System.getenv("MUSIQUE_RESPONSE_TYPE")
                ?: "JSON response (response, score, follow_up_queries)"

        require(parallelism > 0) { "MUSIQUE_PARALLELISM must be positive." }
        require(topKContexts > 0) { "MUSIQUE_TOP_K_CONTEXTS must be positive." }

        val queryConfig =
            QueryConfigLoader.load(
                root = cliArgs.root,
                configPath = cliArgs.configPath,
                overrideOutputDirs = cliArgs.outputDirs,
            )
        val indexData = QueryIndexLoader(queryConfig.indexes).load()

        val configuredChatModel = queryConfig.defaultChatModel?.model ?: "gpt-4o-mini"
        val configuredEmbeddingModel = queryConfig.defaultEmbeddingModel ?: "text-embedding-3-small"
        val chatModelName = System.getenv("MUSIQUE_CHAT_MODEL") ?: configuredChatModel
        val embeddingModelName = System.getenv("MUSIQUE_EMBEDDING_MODEL") ?: configuredEmbeddingModel
        val evalModelName = System.getenv("MUSIQUE_EVAL_MODEL") ?: chatModelName

        val lines =
            Files
                .readAllLines(cliArgs.inputPath)
                .map { it.trim() }
                .filter { it.isNotBlank() }
        if (lines.isEmpty()) {
            System.err.println("No MusiQue samples found in ${cliArgs.inputPath}")
            kotlin.system.exitProcess(1)
        }

        val examples =
            lines
                .map { inputJson.decodeFromString(MusiqueExample.serializer(), it) }
                .filter { it.answerable }
                .let { list -> if (limit != null) list.take(limit) else list }
        if (examples.isEmpty()) {
            System.err.println("No answerable MusiQue samples were processed.")
            kotlin.system.exitProcess(1)
        }

        val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
        val resultsDir = Path.of("eval_results").resolve("musique_graphrag_ragas_$timestamp")
        Files.createDirectories(resultsDir)

        val generated = mutableListOf<GeneratedSample>()
        val processed = AtomicInteger(0)
        val semaphore = Semaphore(parallelism)

        runBlocking {
            val jobs =
                examples.map { example ->
                    async(Dispatchers.IO) {
                        semaphore.withPermit {
                            try {
                                val generatedSample =
                                    generateSample(
                                        example = example,
                                        apiKey = apiKey,
                                        method = method,
                                        queryConfig = queryConfig,
                                        indexData = indexData,
                                        responseType = responseType,
                                        communityLevel = communityLevel,
                                        dynamicCommunitySelection = dynamicCommunitySelection,
                                        topKContexts = topKContexts,
                                        chatModelName = chatModelName,
                                        embeddingModelName = embeddingModelName,
                                    )
                                synchronized(generated) {
                                    generated += generatedSample
                                }

                                val count = processed.incrementAndGet()
                                if (count % 10 == 0 || count == examples.size) {
                                    println("Generated $count / ${examples.size} samples")
                                }
                            } catch (ex: Exception) {
                                System.err.println("Error processing sample ${example.id}: ${ex.message}")
                            }
                        }
                    }
                }
            jobs.awaitAll()
        }

        if (generated.isEmpty()) {
            System.err.println("No MusiQue samples were successfully generated.")
            kotlin.system.exitProcess(1)
        }

        val dataset =
            EvaluationDataset(
                generated.map { sample ->
                    SingleTurnSample(
                        userInput = sample.question,
                        retrievedContexts = sample.retrievedContexts,
                        response = sample.prediction,
                        reference = sample.reference,
                    )
                },
            )

        val evalChatModel =
            OpenAiChatModel
                .builder()
                .apiKey(apiKey)
                .modelName(evalModelName)
                .temperature(0.0)
                .build()
        val ragasLlm =
            LangChain4jLlm(
                model = evalChatModel,
                runConfig = RunConfig(timeoutSeconds = 90),
            )

        val correctnessMetric = AnswerCorrectnessMetric(name = "correctness", weights = listOf(1.0, 0.0))
        val precisionMetric = FactualCorrectnessMetric(name = "precision", mode = FactualCorrectnessMetric.Mode.PRECISION)
        val recallMetric = FactualCorrectnessMetric(name = "recall", mode = FactualCorrectnessMetric.Mode.RECALL)
        val f1Metric = FactualCorrectnessMetric(name = "f1", mode = FactualCorrectnessMetric.Mode.F1)

        val ragasResult =
            evaluate(
                dataset = dataset,
                metrics = listOf(correctnessMetric, precisionMetric, recallMetric, f1Metric),
                llm = ragasLlm,
            )

        val exactMatchTotal = DoubleAdder()
        val correctnessTotal = DoubleAdder()
        val precisionTotal = DoubleAdder()
        val recallTotal = DoubleAdder()
        val f1Total = DoubleAdder()

        generated.forEachIndexed { index, sample ->
            val row = ragasResult.scores.getOrNull(index).orEmpty()
            val correctness = (row["correctness"] as? Number)?.toDouble() ?: Double.NaN
            val precision = (row["precision"] as? Number)?.toDouble() ?: Double.NaN
            val recall = (row["recall"] as? Number)?.toDouble() ?: Double.NaN
            val f1 = (row["f1"] as? Number)?.toDouble() ?: Double.NaN

            exactMatchTotal.add(sample.exactMatch)
            if (!correctness.isNaN()) correctnessTotal.add(correctness)
            if (!precision.isNaN()) precisionTotal.add(precision)
            if (!recall.isNaN()) recallTotal.add(recall)
            if (!f1.isNaN()) f1Total.add(f1)

            val perSample =
                buildString {
                    appendLine("id: ${sample.id}")
                    appendLine("question: ${sample.question}")
                    appendLine("prediction: ${sample.prediction}")
                    appendLine("gold: ${sample.primaryGold}")
                    if (sample.aliases.isNotEmpty()) {
                        appendLine("aliases: ${sample.aliases.joinToString(", ")}")
                    }
                    appendLine("reference_used_for_ragas: ${sample.reference}")
                    appendLine("exact_match: ${sample.exactMatch}")
                    appendLine("correctness: ${"%.4f".format(correctness)}")
                    appendLine("precision: ${"%.4f".format(precision)}")
                    appendLine("recall: ${"%.4f".format(recall)}")
                    appendLine("f1: ${"%.4f".format(f1)}")
                    if (sample.retrievedContexts.isNotEmpty()) {
                        appendLine("retrieved_contexts:")
                        sample.retrievedContexts.forEachIndexed { idx, ctx ->
                            appendLine("  [${idx + 1}] $ctx")
                        }
                    }
                }
            Files.writeString(resultsDir.resolve("${sanitizeSampleId(sample.id)}.txt"), perSample)
        }

        val count = generated.size.toDouble()
        val exactMatch = exactMatchTotal.sum() / count
        val correctness = correctnessTotal.sum() / count
        val precision = precisionTotal.sum() / count
        val recall = recallTotal.sum() / count
        val f1 = f1Total.sum() / count

        println("MusiQue + ragas evaluation completed for ${generated.size} samples")
        println("Pipeline: GraphRAG")
        println("Query Method: ${method.name.lowercase()}")
        println("Chat Model: $chatModelName")
        println("Embedding Model: $embeddingModelName")
        println("Evaluation Model: $evalModelName")
        println("ExactMatch: ${"%.4f".format(exactMatch)}")
        println("Correctness: ${"%.4f".format(correctness)}")
        println("Precision: ${"%.4f".format(precision)}")
        println("Recall: ${"%.4f".format(recall)}")
        println("F1: ${"%.4f".format(f1)}")
        println("Per-sample outputs written to $resultsDir")
    }

    @Suppress("LongMethod")
    private suspend fun generateSample(
        example: MusiqueExample,
        apiKey: String,
        method: EvalQueryMethod,
        queryConfig: QueryConfig,
        indexData: QueryIndexData,
        responseType: String,
        communityLevel: Int,
        dynamicCommunitySelection: Boolean,
        topKContexts: Int,
        chatModelName: String,
        embeddingModelName: String,
    ): GeneratedSample {
        val callbacks = CollectingQueryCallbacks()
        val callbackList = listOf(callbacks)

        fun buildStreamingModel(modelConfig: QueryModelConfig?): OpenAiStreamingChatModel {
            val builder =
                OpenAiStreamingChatModel
                    .builder()
                    .apiKey(apiKey)
                    .modelName(modelConfig?.model ?: chatModelName)
            val params = modelConfig?.params
            params?.temperature?.let { builder.temperature(it) }
            params?.topP?.let { builder.topP(it) }
            params?.maxTokens?.let { builder.maxTokens(it) }
            return builder.build()
        }

        fun createLocalEngine(filteredReports: List<CommunityReport>): LocalQueryEngine {
            val modelConfig = queryConfig.local.chat ?: queryConfig.defaultChatModel
            val localParams = modelConfig?.params ?: ModelParams(jsonResponse = true)
            return LocalQueryEngine(
                streamingModel = buildStreamingModel(modelConfig),
                embeddingModel = defaultEmbeddingModel(apiKey, queryConfig.local.embeddingModel ?: embeddingModelName),
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
                callbacks = callbackList,
            )
        }

        fun createGlobalEngine(filteredReports: List<CommunityReport>): GlobalSearchEngine {
            val modelConfig = queryConfig.global.chat ?: queryConfig.defaultChatModel
            val baseParams = modelConfig?.params ?: ModelParams()
            val mapParams = baseParams.copy(jsonResponse = true)
            val reduceParams = baseParams.copy(jsonResponse = false)
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
                callbacks = callbackList,
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

        val filteredReports = filterCommunityReports(indexData.communityReports, indexData.communityHierarchy, communityLevel)
        val result: QueryResult =
            when (method) {
                EvalQueryMethod.BASIC -> {
                    val modelConfig = queryConfig.basic.chat ?: queryConfig.defaultChatModel
                    val engine =
                        BasicQueryEngine(
                            streamingModel = buildStreamingModel(modelConfig),
                            embeddingModel = defaultEmbeddingModel(apiKey, queryConfig.basic.embeddingModel ?: embeddingModelName),
                            vectorStore = indexData.vectorStore,
                            textUnits = indexData.textUnits,
                            textEmbeddings = indexData.textEmbeddings,
                            topK = queryConfig.basic.k,
                            maxContextTokens = queryConfig.basic.maxContextTokens,
                            callbacks = callbackList,
                            systemPrompt = queryConfig.basic.prompt,
                        )
                    engine.answer(example.question, responseType)
                }

                EvalQueryMethod.LOCAL -> {
                    val localEngine = createLocalEngine(filteredReports)
                    localEngine.answer(example.question, responseType)
                }

                EvalQueryMethod.GLOBAL -> {
                    val globalEngine = createGlobalEngine(filteredReports)
                    val globalResult = globalEngine.search(example.question)
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

                EvalQueryMethod.DRIFT -> {
                    val localEngine = createLocalEngine(filteredReports)
                    val globalEngine = createGlobalEngine(filteredReports)
                    val driftModel = queryConfig.drift.chat ?: queryConfig.defaultChatModel
                    val driftEngine =
                        DriftSearchEngine(
                            streamingModel = buildStreamingModel(driftModel),
                            communityReports = filteredReports,
                            globalSearchEngine = globalEngine,
                            localQueryEngine = localEngine,
                            primerSystemPrompt = queryConfig.drift.prompt,
                            reduceSystemPrompt = queryConfig.drift.reducePrompt,
                            callbacks = callbackList,
                            maxIterations = queryConfig.drift.maxIterations,
                        )
                    val driftResult = driftEngine.search(example.question)
                    QueryResult(
                        answer = driftResult.answer,
                        context = emptyList(),
                        contextRecords = callbacks.contextRecords,
                        contextText = callbacks.reduceContext,
                        llmCalls = driftResult.llmCalls,
                        promptTokens = driftResult.promptTokens,
                        outputTokens = driftResult.outputTokens,
                        llmCallsCategories = driftResult.llmCallsCategories,
                        promptTokensCategories = driftResult.promptTokensCategories,
                        outputTokensCategories = driftResult.outputTokensCategories,
                    )
                }
            }

        val prediction = extractPredictionText(result.answer)
        val retrievedContexts = extractRetrievedContexts(result.context.map { it.text }, result.contextRecords).take(topKContexts)
        val golds = listOf(example.answer) + example.answerAliases
        val reference = pickBestReferenceForPrediction(prediction, golds)
        val em = bestExactMatch(prediction, golds)
        return GeneratedSample(
            id = example.id,
            question = example.question,
            prediction = prediction,
            reference = reference,
            primaryGold = example.answer,
            aliases = example.answerAliases,
            retrievedContexts = retrievedContexts,
            exactMatch = em,
        )
    }

    private fun parseCliArgs(args: Array<String>): CliArgs {
        var inputPath: Path = Path.of("data/musique_ans_v1.0_train-200.jsonl")
        var root: Path = Path.of(".")
        var configPath: Path? = null
        val dataDirs = mutableListOf<Path>()

        var i = 0
        while (i < args.size) {
            when (val arg = args[i]) {
                "--input" -> {
                    val value = args.getOrNull(i + 1) ?: usageAndExit("Missing value for --input")
                    inputPath = Path.of(value)
                    i += 2
                }

                "--root" -> {
                    val value = args.getOrNull(i + 1) ?: usageAndExit("Missing value for --root")
                    root = Path.of(value)
                    i += 2
                }

                "--config" -> {
                    val value = args.getOrNull(i + 1) ?: usageAndExit("Missing value for --config")
                    configPath = Path.of(value)
                    i += 2
                }

                "--data" -> {
                    val value = args.getOrNull(i + 1) ?: usageAndExit("Missing value for --data")
                    dataDirs += value.split(',').map { Path.of(it.trim()) }.filter { it.toString().isNotBlank() }
                    i += 2
                }

                else -> {
                    if (arg.startsWith("--")) {
                        usageAndExit("Unknown option: $arg")
                    }
                    inputPath = Path.of(arg)
                    i += 1
                }
            }
        }

        return CliArgs(inputPath = inputPath, root = root, configPath = configPath, outputDirs = dataDirs)
    }

    private fun parseMethod(raw: String?): EvalQueryMethod {
        val method = raw?.lowercase()?.trim().orEmpty()
        return when (method) {
            "", "local" -> {
                EvalQueryMethod.LOCAL
            }

            "basic" -> {
                EvalQueryMethod.BASIC
            }

            "global" -> {
                EvalQueryMethod.GLOBAL
            }

            "drift" -> {
                EvalQueryMethod.DRIFT
            }

            else -> {
                System.err.println("Unknown MUSIQUE_QUERY_METHOD '$raw'; defaulting to local")
                EvalQueryMethod.LOCAL
            }
        }
    }

    private fun extractPredictionText(rawAnswer: String): String {
        val trimmed = rawAnswer.trim()
        val jsonText =
            if (trimmed.startsWith("```") && trimmed.endsWith("```")) {
                trimmed
                    .removePrefix("```")
                    .removePrefix("json")
                    .removeSuffix("```")
                    .trim()
            } else {
                trimmed
            }

        val parsed =
            runCatching {
                inputJson.parseToJsonElement(jsonText) as? JsonObject
            }.getOrNull()
        val response = parsed?.get("response")?.jsonPrimitive?.contentOrNull
        return (response ?: trimmed).trim()
    }

    private fun extractRetrievedContexts(
        contextChunks: List<String>,
        contextRecords: Map<String, List<Map<String, String>>>,
    ): List<String> {
        if (contextChunks.isNotEmpty()) {
            return contextChunks.mapNotNull { it.trim().takeIf { text -> text.isNotBlank() } }
        }

        val candidateKeys = listOf("text", "content", "summary", "description", "claim", "full_content")
        return contextRecords.values
            .flatten()
            .mapNotNull { row ->
                candidateKeys.firstNotNullOfOrNull { key -> row[key]?.trim()?.takeIf { it.isNotBlank() } }
            }.distinct()
    }

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

    private fun usageAndExit(message: String): Nothing {
        System.err.println(message)
        System.err.println(
            "Usage: MusiQueGraphRagRagas [--input <path>] [--root <path>] [--config <path>] [--data <dir1,dir2,...>] [<input_path>]",
        )
        kotlin.system.exitProcess(1)
    }

    private fun sanitizeSampleId(sampleId: String): String = sampleId.replace(Regex("[^A-Za-z0-9._-]"), "_")

    private fun pickBestReferenceForPrediction(
        prediction: String,
        golds: List<String>,
    ): String =
        golds
            .maxByOrNull { candidate ->
                if (normalize(prediction) == normalize(candidate)) {
                    10_000
                } else {
                    tokenOverlapScore(prediction, candidate)
                }
            } ?: ""

    private fun tokenOverlapScore(
        prediction: String,
        gold: String,
    ): Int {
        val predTokens = tokenize(normalize(prediction))
        val goldTokens = tokenize(normalize(gold))
        if (predTokens.isEmpty() || goldTokens.isEmpty()) return 0
        val predCounts = predTokens.groupingBy { it }.eachCount()
        val goldCounts = goldTokens.groupingBy { it }.eachCount()
        var overlap = 0
        for ((token, pCount) in predCounts) {
            val gCount = goldCounts[token] ?: 0
            overlap += min(pCount, gCount)
        }
        return overlap
    }

    private fun bestExactMatch(
        prediction: String,
        golds: List<String>,
    ): Double = golds.maxOfOrNull { if (normalize(prediction) == normalize(it)) 1.0 else 0.0 } ?: 0.0

    private fun normalize(text: String): String {
        val lowered = text.lowercase()
        val noPunc = lowered.replace(Regex("[^a-z0-9\\s]"), " ")
        val noArticles = noPunc.replace(Regex("\\b(a|an|the)\\b"), " ")
        return noArticles.replace(Regex("\\s+"), " ").trim()
    }

    private fun tokenize(text: String): List<String> =
        if (text.isBlank()) {
            emptyList()
        } else {
            text.split(' ')
        }
}
