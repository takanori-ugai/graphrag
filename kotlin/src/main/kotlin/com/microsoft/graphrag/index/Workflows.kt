package com.microsoft.graphrag.index

import dev.langchain4j.model.chat.Capability.RESPONSE_FORMAT_JSON_SCHEMA
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.time.Instant

/**
 * Realistic workflow skeleton: load documents, chunk, extract entities/relationships with LLM,
 * embed text/entities, build graph, write parquet outputs. Community steps remain TODO.
 */
fun defaultPipeline(): Pipeline {
    validateApiKey()
    return SimplePipeline(
        listOf(
            "load_input_documents" to ::loadInputDocuments,
            "extract_graph" to ::extractGraph,
            "extract_claims" to ::extractClaims,
            "summarize_descriptions" to ::summarizeDescriptions,
            "embed_text" to ::embedText,
            "embed_graph" to ::embedGraph,
            "build_graph" to ::buildGraph,
            "community_detection" to ::communityDetection,
            "community_reports" to ::communityReports,
            "write_outputs" to ::writeOutputs,
        ),
    )
}

private fun validateApiKey() {
    // Force initialization to fail fast if OPENAI_API_KEY is missing.
    openAiApiKey
}

private val openAiApiKey: String by lazy { defaultApiKey() }

private val sharedChatModel: dev.langchain4j.model.openai.OpenAiChatModel by lazy { defaultChatModel() }

private suspend fun loadInputDocuments(
    config: GraphRagConfig,
    context: PipelineRunContext,
): WorkflowResult {
    val chunker = DocumentChunker()
    val chunks = chunker.loadAndChunk(config.inputDir)
    context.state["chunks"] = chunks
    context.state["text_units"] = chunker.toTextUnits(chunks)
    context.outputStorage.set("documents_loaded.txt", "Loaded ${chunks.size} chunks at ${Instant.now()}")
    return WorkflowResult(result = "documents")
}

@Suppress("UnusedParameter")
private suspend fun extractGraph(
    config: GraphRagConfig,
    context: PipelineRunContext,
): WorkflowResult {
    val chunks = context.state["chunks"] as? List<DocumentChunk> ?: emptyList()
    val chatModel = sharedChatModel

    val extractor = ExtractGraphWorkflow(chatModel)
    val result = extractor.extract(chunks)
    context.state["entities"] = result.entities
    context.state["relationships"] = result.relationships
    context.outputStorage.set(
        "graph_extracted.txt",
        "Extracted ${result.entities.size} entities and ${result.relationships.size} relationships at ${Instant.now()}",
    )
    return WorkflowResult(result = "graph")
}

@Suppress("UnusedParameter")
private suspend fun extractClaims(
    config: GraphRagConfig,
    context: PipelineRunContext,
): WorkflowResult {
    val chunks = context.state["chunks"] as? List<DocumentChunk> ?: emptyList()
    val chatModel = sharedChatModel
    val claimsWorkflow = ClaimsWorkflow(chatModel)
    val claims = claimsWorkflow.extractClaims(chunks)
    context.state["claims"] = claims
    context.outputStorage.set(
        "claims_extracted.txt",
        "Extracted ${claims.size} claims at ${Instant.now()}",
    )
    return WorkflowResult(result = "claims")
}

@Suppress("UnusedParameter")
private suspend fun embedText(
    config: GraphRagConfig,
    context: PipelineRunContext,
): WorkflowResult {
    val chunks = context.state["chunks"] as? List<DocumentChunk> ?: emptyList()
    val embedder = EmbedWorkflow(defaultEmbeddingModel(openAiApiKey))
    val textEmbeddings =
        embedder.embedChunks(
            chunks,
            progress = { context.callbacks.progress(it) },
            description = "Text embedding progress: ",
        )
    context.state["text_embeddings"] = textEmbeddings
    context.outputStorage.set(
        "text_embeddings.txt",
        "Text embeddings generated for ${textEmbeddings.size} chunks at ${Instant.now()}",
    )
    return WorkflowResult(result = "text_embeddings")
}

@Suppress("UnusedParameter")
private suspend fun embedGraph(
    config: GraphRagConfig,
    context: PipelineRunContext,
): WorkflowResult {
    val entities = context.state["entities"] as? List<Entity> ?: emptyList()
    val embedder = EmbedWorkflow(defaultEmbeddingModel(openAiApiKey))
    val entityEmbeddings =
        embedder.embedEntities(
            entities,
            progress = { context.callbacks.progress(it) },
            description = "Entity embedding progress: ",
        )
    context.state["entity_embeddings"] = entityEmbeddings
    context.outputStorage.set(
        "graph_embeddings.txt",
        "Entity embeddings generated for ${entityEmbeddings.size} entities at ${Instant.now()}",
    )
    return WorkflowResult(result = "entity_embeddings")
}

@Suppress("UnusedParameter")
private suspend fun summarizeDescriptions(
    config: GraphRagConfig,
    context: PipelineRunContext,
): WorkflowResult {
    val entities = context.state["entities"] as? List<Entity> ?: emptyList()
    val relationships = context.state["relationships"] as? List<Relationship> ?: emptyList()
    val chatModel = sharedChatModel
    val summarizer = SummarizeDescriptionsWorkflow(chatModel)
    val textUnits = context.state["text_units"] as? List<TextUnit> ?: emptyList()
    val summaries = summarizer.summarize(entities, relationships, textUnits)
    context.state["entity_summaries"] = summaries.entitySummaries
    context.state["relationships"] = summaries.relationships
    context.outputStorage.set(
        "entity_summaries.txt",
        "Summarized ${summaries.entitySummaries.size} entities at ${Instant.now()}",
    )
    return WorkflowResult(result = "entity_summaries")
}

@Suppress("UnusedParameter")
private suspend fun buildGraph(
    config: GraphRagConfig,
    context: PipelineRunContext,
): WorkflowResult {
    val entities = context.state["entities"] as? List<Entity> ?: emptyList()
    val relationships = context.state["relationships"] as? List<Relationship> ?: emptyList()
    val graph = GraphBuilder().buildGraph(entities, relationships)
    context.state["graph"] = graph
    context.outputStorage.set(
        "graph_built.txt",
        "Graph constructed with ${graph.vertexSet().size} nodes at ${Instant.now()}",
    )
    return WorkflowResult(result = "graph_built")
}

@Suppress("UnusedParameter")
private suspend fun writeOutputs(
    config: GraphRagConfig,
    context: PipelineRunContext,
): WorkflowResult {
    val entities = context.state["entities"] as? List<Entity> ?: emptyList()
    val relationships = context.state["relationships"] as? List<Relationship> ?: emptyList()
    val writer = ParquetWriterHelper()
    val textEmbeddings = context.state["text_embeddings"] as? List<TextEmbedding> ?: emptyList()
    val entityEmbeddings = context.state["entity_embeddings"] as? List<EntityEmbedding> ?: emptyList()
    val communities = context.state["communities"] as? List<CommunityAssignment> ?: emptyList()
    val reports = context.state["community_reports"] as? List<CommunityReport> ?: emptyList()
    val claims = context.state["claims"] as? List<Claim> ?: emptyList()
    val textUnits = context.state["text_units"] as? List<TextUnit> ?: emptyList()
    val summaries = context.state["entity_summaries"] as? List<EntitySummary> ?: emptyList()
    val reportsJson =
        kotlinx.serialization.json
            .Json { prettyPrint = true }
            .encodeToString(
                kotlinx.serialization.builtins.ListSerializer(CommunityReport.serializer()),
                reports,
            )
    withContext(Dispatchers.IO) {
        writer.writeEntities(config.outputDir.resolve("entities.parquet"), entities)
        writer.writeRelationships(config.outputDir.resolve("relationships.parquet"), relationships)
        writer.writeTextEmbeddings(config.outputDir.resolve("text_embeddings.parquet"), textEmbeddings)
        writer.writeEntityEmbeddings(config.outputDir.resolve("entity_embeddings.parquet"), entityEmbeddings)
        writer.writeCommunityAssignments(config.outputDir.resolve("communities.parquet"), communities)
        config.outputDir.resolve("community_reports.json").apply {
            java.nio.file.Files
                .createDirectories(parent)
            java.nio.file.Files
                .writeString(this, reportsJson)
        }
        writer.writeTextUnits(config.outputDir.resolve("text_units.parquet"), textUnits)
        writer.writeEntitySummaries(config.outputDir.resolve("entity_summaries.parquet"), summaries)
        writer.writeClaims(config.outputDir.resolve("claims.parquet"), claims)
        val vectorStore = LocalVectorStore(config.outputDir.resolve("vector_store.json"))
        vectorStore.save(textEmbeddings, entityEmbeddings)
    }
    context.outputStorage.set("outputs_written.txt", "Parquet written at ${Instant.now()}")
    return WorkflowResult(result = "outputs_written")
}

@Suppress("UnusedParameter")
private suspend fun communityDetection(
    config: GraphRagConfig,
    context: PipelineRunContext,
): WorkflowResult {
    val entities = context.state["entities"] as? List<Entity> ?: emptyList()
    val relationships = context.state["relationships"] as? List<Relationship> ?: emptyList()
    val detector = CommunityDetectionWorkflow()
    val detection = detector.detect(entities, relationships)
    val assignments = detection.assignments
    context.state["communities"] = assignments
    context.state["community_hierarchy"] = detection.hierarchy
    context.outputStorage.set(
        "communities.txt",
        "Detected ${assignments.map { it.communityId }.distinct().size} communities at ${Instant.now()}",
    )
    return WorkflowResult(result = "communities")
}

@Suppress("UnusedParameter")
private suspend fun communityReports(
    config: GraphRagConfig,
    context: PipelineRunContext,
): WorkflowResult {
    val communities = context.state["communities"] as? List<CommunityAssignment> ?: emptyList()
    val entities = context.state["entities"] as? List<Entity> ?: emptyList()
    val relationships = context.state["relationships"] as? List<Relationship> ?: emptyList()
    val textUnits = context.state["text_units"] as? List<TextUnit> ?: emptyList()
    val claims = context.state["claims"] as? List<Claim> ?: emptyList()
    val hierarchy =
        (context.state["community_hierarchy"] as? Map<*, *> ?: emptyMap<Any, Any>())
            .mapNotNull { (k, v) ->
                val key = k as? Int
                val value = v as? Int
                if (key != null) key to value else null
            }.toMap()
    val priorReports = context.state["community_reports"] as? List<CommunityReport> ?: emptyList()
    val chatModel = sharedChatModel
    val reporter = CommunityReportWorkflow(chatModel)
    val reports =
        reporter.generateReports(
            communities,
            entities,
            relationships,
            textUnits,
            claims,
            hierarchy,
            priorReports,
        )
    context.state["community_reports"] = reports
    context.outputStorage.set(
        "community_reports.txt",
        "Generated ${reports.size} community reports at ${Instant.now()}",
    )
    return WorkflowResult(result = "community_reports")
}

private fun defaultChatModel(): dev.langchain4j.model.openai.OpenAiChatModel =
    dev.langchain4j.model.openai.OpenAiChatModel
        .builder()
        .apiKey(openAiApiKey)
        .modelName("gpt-4o-mini")
        .supportedCapabilities(RESPONSE_FORMAT_JSON_SCHEMA)
        .strictJsonSchema(true)
        .build()

private fun defaultApiKey(): String =
    System.getenv("OPENAI_API_KEY")
        ?: error("OPENAI_API_KEY environment variable is required")
