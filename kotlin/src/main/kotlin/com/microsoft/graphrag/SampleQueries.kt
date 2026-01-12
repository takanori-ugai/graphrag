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
 * Small sample that runs basic, local, global, and drift queries against the sample-index.
 *
 * Run with:
 * ./gradlew run -PmainClass=SampleQueriesKt
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
        val globalResult =
            GlobalSearchEngine(
                streamingModel = streamingModel,
                communityReports = index.communityReports,
            ).search(question)
        println("\n=== GLOBAL ===")
        println(globalResult.answer)

        // DRIFT
        val driftResult =
            DriftSearchEngine(
                streamingModel = streamingModel,
                communityReports = index.communityReports,
                globalSearchEngine =
                    GlobalSearchEngine(
                        streamingModel = streamingModel,
                        communityReports = index.communityReports,
                    ),
                localQueryEngine = localEngine,
            ).search(question)
        println("\n=== DRIFT ===")
        println(driftResult.answer)
    }
