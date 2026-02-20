package com.microsoft.graphrag

import com.microsoft.graphrag.evaluation.QAExactMatch
import com.microsoft.graphrag.evaluation.QAF1Score
import com.microsoft.graphrag.index.DocumentChunk
import com.microsoft.graphrag.index.EmbedWorkflow
import com.microsoft.graphrag.index.LocalVectorStore
import com.microsoft.graphrag.index.Payload
import com.microsoft.graphrag.index.TextUnit
import com.microsoft.graphrag.index.defaultEmbeddingModel
import com.microsoft.graphrag.query.BasicQueryEngine
import com.microsoft.graphrag.query.JsonAnswerParser
import dev.langchain4j.model.openai.OpenAiStreamingChatModel
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.nio.file.Path
import kotlin.system.exitProcess

/**
 * CLI entry point for running and evaluating QA over MuSiQue JSONL samples.
 */
fun main(args: Array<String>) =
    runBlocking {
        val parsed = MusiqueArgs.parse(args)
        val apiKey = System.getenv("OPENAI_API_KEY")
        if (apiKey.isNullOrBlank()) {
            println("OPENAI_API_KEY is not set. Please set it to run the evaluation.")
            return@runBlocking
        }

        val samples = readMusiqueSamples(parsed.inputPath, parsed.limit)
        if (samples.isEmpty()) {
            println("No samples found.")
            return@runBlocking
        }

        val streamingModel =
            OpenAiStreamingChatModel
                .builder()
                .apiKey(apiKey)
                .modelName(parsed.llmName ?: "gpt-4o-mini")
                .build()
        val embeddingModel = defaultEmbeddingModel(apiKey, parsed.embeddingName ?: "text-embedding-3-small")
        val embedder = EmbedWorkflow(embeddingModel)

        val predictions = mutableListOf<String>()
        val goldAnswers = mutableListOf<List<String>>()

        samples.forEachIndexed { index, sample ->
            val docs =
                sample.paragraphs
                    .map { paragraph ->
                        val title = paragraph.title?.trim().orEmpty()
                        if (title.isNotEmpty()) {
                            "$title\n${paragraph.paragraphText}"
                        } else {
                            paragraph.paragraphText
                        }
                    }.filter { it.isNotBlank() }

            if (docs.isEmpty()) {
                println("Skipping sample ${sample.id}: no non-blank passages.")
                predictions.add("")
                val gold =
                    buildList {
                        add(sample.answer)
                        addAll(sample.answerAliases)
                    }.distinct()
                goldAnswers.add(gold)
                return@forEachIndexed
            }

            val chunks =
                docs.mapIndexed { idx, text ->
                    DocumentChunk(
                        id = "chunk-${sample.id}-$idx",
                        sourcePath = "musique-${sample.id}",
                        text = text,
                    )
                }
            val textUnits =
                chunks.map { chunk ->
                    TextUnit(
                        id = chunk.id,
                        chunkId = chunk.id,
                        text = chunk.text,
                        sourcePath = chunk.sourcePath,
                    )
                }
            val textEmbeddings = embedder.embedChunks(chunks)
            val vectorStore =
                LocalVectorStore(
                    path = Path.of("in-memory", sample.id),
                    payloadOverride = Payload(textEmbeddings, emptyList()),
                )

            val result =
                BasicQueryEngine(
                    streamingModel = streamingModel,
                    embeddingModel = embeddingModel,
                    vectorStore = vectorStore,
                    textUnits = textUnits,
                    textEmbeddings = textEmbeddings,
                    topK = parsed.topK,
                ).answer(
                    question = sample.question,
                    responseType = "Answer with one or a few words only (no sentences).",
                )

            val normalized = normalizeModelResponse(result.answer)
            val answer = JsonAnswerParser.parse(normalized).response.trim()
            predictions.add(answer)

            val gold =
                buildList {
                    add(sample.answer)
                    addAll(sample.answerAliases)
                }.distinct()
            goldAnswers.add(gold)
            println("Answer: $answer")
            println("Gold: $gold")

            if ((index + 1) % 10 == 0 || index == samples.lastIndex) {
                println("Processed ${index + 1}/${samples.size} samples.")
            }
        }

        val emEvaluator = QAExactMatch()
        val f1Evaluator = QAF1Score()
        val (overallEm, _) =
            emEvaluator.calculateMetricScores(
                goldAnswers = goldAnswers,
                predictedAnswers = predictions,
                aggregationFn = { values -> values.maxOrNull() ?: 0.0 },
            )
        val (overallF1, _) =
            f1Evaluator.calculateMetricScores(
                goldAnswers = goldAnswers,
                predictedAnswers = predictions,
                aggregationFn = { values -> values.maxOrNull() ?: 0.0 },
            )

        println("=== Evaluation ===")
        println("ExactMatch: ${"%.4f".format(overallEm.getValue("ExactMatch"))}")
        println("F1: ${"%.4f".format(overallF1.getValue("F1"))}")
    }

private fun readMusiqueSamples(
    path: String,
    limit: Int?,
): List<MusiqueSample> {
    val json = Json { ignoreUnknownKeys = true }
    val file = resolveMusiqueFile(path, "input")
    val lines = file.readLines().filter { it.isNotBlank() }
    val capped = if (limit != null) lines.take(limit) else lines
    return capped.map { line -> json.decodeFromString(MusiqueSample.serializer(), line) }
}

private fun resolveMusiqueFile(
    path: String,
    label: String,
): File {
    val baseDir = File(".").canonicalFile
    val file = File(path).canonicalFile
    val basePath = baseDir.path + File.separator
    require(file.path.startsWith(basePath)) {
        "Refusing to read $label outside working directory: ${file.path}"
    }
    require(file.isFile) {
        "Missing or invalid $label file: ${file.path}"
    }
    return file
}

private fun normalizeModelResponse(raw: String): String {
    val trimmed = raw.trim()
    if (trimmed.startsWith("```")) {
        val withoutFenceHeader = trimmed.substringAfter('\n', "")
        val withoutFence = withoutFenceHeader.substringBeforeLast("```", withoutFenceHeader)
        return withoutFence.trim()
    }
    val start = trimmed.indexOf('{')
    val end = trimmed.lastIndexOf('}')
    return if (start >= 0 && end > start) trimmed.substring(start, end + 1).trim() else trimmed
}

@Serializable
private data class MusiqueParagraph(
    val idx: Int? = null,
    val title: String? = null,
    @SerialName("paragraph_text")
    val paragraphText: String,
    @SerialName("is_supporting")
    val isSupporting: Boolean? = null,
)

@Serializable
private data class MusiqueSample(
    val id: String,
    val paragraphs: List<MusiqueParagraph>,
    val question: String,
    val answer: String,
    @SerialName("answer_aliases")
    val answerAliases: List<String> = emptyList(),
    val answerable: Boolean? = null,
)

private data class MusiqueArgs(
    val inputPath: String,
    val limit: Int?,
    val llmName: String?,
    val embeddingName: String?,
    val topK: Int,
) {
    companion object {
        fun parse(args: Array<String>): MusiqueArgs {
            val map = mutableMapOf<String, String>()
            var i = 0
            while (i < args.size) {
                val key = args[i]
                if (!key.startsWith("--") || i + 1 >= args.size) {
                    val message = if (key.startsWith("--")) "Missing value for $key." else "Unexpected argument: $key."
                    usageAndExit(message)
                }
                map[key.removePrefix("--")] = args[i + 1]
                i += 2
            }

            val input = map["input"] ?: usageAndExit("Missing required --input.")
            val limit = map["limit"]?.toIntOrNull()
            val topK = map["top_k"]?.toIntOrNull() ?: 10

            return MusiqueArgs(
                inputPath = input,
                limit = limit,
                llmName = map["llm_name"],
                embeddingName = map["embedding_name"],
                topK = topK,
            )
        }

        private fun usageAndExit(reason: String? = null): Nothing {
            if (reason != null) {
                System.err.println(reason)
            }
            System.err.println(
                "Usage: --input <musique.jsonl> [--limit <n>] [--llm_name gpt-4o-mini] " +
                    "[--embedding_name text-embedding-3-small] [--top_k 10]\n" +
                    "Note: file paths must resolve within the current working directory.",
            )
            exitProcess(2)
        }
    }
}
