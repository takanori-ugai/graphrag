package com.microsoft.graphrag.index

import dev.langchain4j.model.openai.OpenAiChatModel
import java.time.Instant

/**
 * Realistic workflow skeleton: load documents, chunk, extract entities/relationships with LLM,
 * embed text/entities, build graph, write parquet outputs. Community steps remain TODO.
 */
fun defaultPipeline(): Pipeline =
    SimplePipeline(
        listOf(
            "load_input_documents" to ::loadInputDocuments,
            "extract_graph" to ::extractGraph,
            "embed_text" to ::embedText,
            "embed_graph" to ::embedGraph,
            "build_graph" to ::buildGraph,
            "community_detection" to ::communityDetection,
            "community_reports" to ::communityReports,
            "write_outputs" to ::writeOutputs,
        ),
    )

private suspend fun loadInputDocuments(
    config: GraphRagConfig,
    context: PipelineRunContext,
): WorkflowResult {
    val chunker = DocumentChunker()
    val chunks = chunker.loadAndChunk(config.inputDir)
    context.state["chunks"] = chunks
    context.outputStorage.set("documents_loaded.txt", "Loaded ${chunks.size} chunks at ${Instant.now()}")
    return WorkflowResult(result = "documents")
}

@Suppress("UnusedParameter")
private suspend fun extractGraph(
    config: GraphRagConfig,
    context: PipelineRunContext,
): WorkflowResult {
    val chunks = context.state["chunks"] as? List<DocumentChunk> ?: emptyList()
    val apiKey = System.getenv("OPENAI_API_KEY") ?: ""
    val chatModel =
        OpenAiChatModel
            .builder()
            .apiKey(apiKey)
            .modelName("gpt-4o-mini")
            .build()

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
private suspend fun embedText(
    config: GraphRagConfig,
    context: PipelineRunContext,
): WorkflowResult {
    val chunks = context.state["chunks"] as? List<DocumentChunk> ?: emptyList()
    val apiKey = System.getenv("OPENAI_API_KEY") ?: ""
    val embedder = EmbedWorkflow(defaultEmbeddingModel(apiKey))
    val textEmbeddings = embedder.embedChunks(chunks)
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
    val apiKey = System.getenv("OPENAI_API_KEY") ?: ""
    val embedder = EmbedWorkflow(defaultEmbeddingModel(apiKey))
    val entityEmbeddings = embedder.embedEntities(entities)
    context.state["entity_embeddings"] = entityEmbeddings
    context.outputStorage.set(
        "graph_embeddings.txt",
        "Entity embeddings generated for ${entityEmbeddings.size} entities at ${Instant.now()}",
    )
    return WorkflowResult(result = "entity_embeddings")
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
    writer.writeEntities(config.outputDir.resolve("entities.parquet"), entities)
    writer.writeRelationships(config.outputDir.resolve("relationships.parquet"), relationships)
    val textEmbeddings = context.state["text_embeddings"] as? List<TextEmbedding> ?: emptyList()
    val entityEmbeddings = context.state["entity_embeddings"] as? List<EntityEmbedding> ?: emptyList()
    writer.writeTextEmbeddings(config.outputDir.resolve("text_embeddings.parquet"), textEmbeddings)
    writer.writeEntityEmbeddings(config.outputDir.resolve("entity_embeddings.parquet"), entityEmbeddings)
    val communities = context.state["communities"] as? List<CommunityAssignment> ?: emptyList()
    writer.writeCommunityAssignments(config.outputDir.resolve("communities.parquet"), communities)
    val reports = context.state["community_reports"] as? List<CommunityReport> ?: emptyList()
    val reportsJson =
        kotlinx.serialization.json
            .Json { prettyPrint = true }
            .encodeToString(
                kotlinx.serialization.builtins.ListSerializer(CommunityReport.serializer()),
                reports,
            )
    config.outputDir.resolve("community_reports.json").apply {
        java.nio.file.Files
            .createDirectories(parent)
        java.nio.file.Files
            .writeString(this, reportsJson)
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
    val assignments = detector.detect(entities, relationships)
    context.state["communities"] = assignments
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
    val apiKey = System.getenv("OPENAI_API_KEY") ?: ""
    val chatModel =
        OpenAiChatModel
            .builder()
            .apiKey(apiKey)
            .modelName("gpt-4o-mini")
            .build()
    val reporter = CommunityReportWorkflow(chatModel)
    val reports = reporter.generateReports(communities, entities)
    context.state["community_reports"] = reports
    context.outputStorage.set(
        "community_reports.txt",
        "Generated ${reports.size} community reports at ${Instant.now()}",
    )
    return WorkflowResult(result = "community_reports")
}
