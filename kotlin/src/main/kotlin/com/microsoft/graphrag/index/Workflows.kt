package com.microsoft.graphrag.index

import dev.langchain4j.model.openai.OpenAiChatModel
import java.time.Instant

/**
 * Realistic workflow skeleton: load documents, chunk, extract entities/relationships with LLM,
 * build graph, write parquet outputs. Embeddings/community steps remain TODO.
 */
fun defaultPipeline(): Pipeline =
    SimplePipeline(
        listOf(
            "load_input_documents" to ::loadInputDocuments,
            "extract_graph" to ::extractGraph,
            "build_graph" to ::buildGraph,
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
    context.outputStorage.set("outputs_written.txt", "Parquet written at ${Instant.now()}")
    return WorkflowResult(result = "outputs_written")
}
