package com.microsoft.graphrag

import com.microsoft.graphrag.index.defaultEmbeddingModel
import com.microsoft.graphrag.query.BasicQueryEngine
import com.microsoft.graphrag.query.DriftSearchEngine
import com.microsoft.graphrag.query.GlobalSearchEngine
import com.microsoft.graphrag.query.LocalQueryEngine
import com.microsoft.graphrag.query.QueryIndexLoader
import dev.langchain4j.model.openai.OpenAiStreamingChatModel
import kotlinx.coroutines.runBlocking
import java.nio.file.Path

/**
 * Runs four sample query scenarios (Basic, Local, Global, Drift) against a prebuilt sample index and prints
 * their answers.
 *
 * Reads OPENAI_API_KEY from the environment, builds an OpenAI streaming chat model and an embedding model,
 * loads the index from
 * "sample-index/output", executes each query engine with a fixed question and response type, and prints each
 * engine's resulting answer to stdout.
 */
fun main() =
    runBlocking {
        val apiKey = System.getenv("OPENAI_API_KEY") ?: error("Set OPENAI_API_KEY")
        val streamingModel =
            OpenAiStreamingChatModel
                .builder()
                .apiKey(apiKey)
                .modelName("gpt-4o-mini")
                .build()
        val embeddingModel = defaultEmbeddingModel(apiKey)

        // Load the sample index produced by the Kotlin indexer
        val indexDir = Path.of("sample-index/output")
        val index = QueryIndexLoader(indexDir).load()

        val question = "Summarize key entities and relationships in the sample index."
        val responseType = "multiple paragraphs"

        // Basic
        val basicResult =
            BasicQueryEngine(
                streamingModel = streamingModel,
                embeddingModel = embeddingModel,
                vectorStore = index.vectorStore,
                textUnits = index.textUnits,
                textEmbeddings = index.textEmbeddings,
                topK = 5,
            ).answer(question, responseType)
        println("=== BASIC ===")
        println(basicResult.answer)

        // Local
        val localEngine =
            LocalQueryEngine(
                streamingModel = streamingModel,
                embeddingModel = embeddingModel,
                vectorStore = index.vectorStore,
                textUnits = index.textUnits,
                textEmbeddings = index.textEmbeddings,
                entities = index.entities,
                entitySummaries = index.entitySummaries,
                relationships = index.relationships,
                claims = index.claims,
                covariates = index.covariates,
                communities = index.communities,
                communityReports = index.communityReports,
                modelParams =
                    com.microsoft.graphrag.query
                        .ModelParams(jsonResponse = false),
            )
        val localResult = localEngine.answer(question, responseType)
        println("\n=== LOCAL ===")
        println(localResult.answer)

        // Global
        val globalEngine =
            GlobalSearchEngine(
                streamingModel = streamingModel,
                communityReports = index.communityReports,
            )
        val globalResult = globalEngine.search(question)
        println("\n=== GLOBAL ===")
        println(globalResult.answer)

        // DRIFT
        val driftResult =
            DriftSearchEngine(
                streamingModel = streamingModel,
                communityReports = index.communityReports,
                globalSearchEngine = globalEngine,
                localQueryEngine = localEngine,
            ).search(question)
        println("\n=== DRIFT ===")
        println(driftResult.answer)
    }
