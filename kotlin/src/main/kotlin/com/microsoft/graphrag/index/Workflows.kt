package com.microsoft.graphrag.index

import dev.langchain4j.model.chat.Capability.RESPONSE_FORMAT_JSON_SCHEMA
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.time.Instant

/**
 * Realistic workflow skeleton: load documents, chunk, extract entities/relationships with LLM,
 * embed text/entities, build graph, write parquet outputs. Community steps remain TODO.
 *
 * Creates the default ordered pipeline for the graph-based RAG index workflow.
 *
 * This function ensures the OpenAI API key is initialized before constructing the pipeline.
 *
 * @return A Pipeline composed of the following ordered steps:
 *   "load_input_documents", "extract_graph", "extract_claims", "summarize_descriptions",
 *   "embed_text", "embed_graph", "build_graph", "community_detection",
 *   "community_reports", and "write_outputs".
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

/**
 * Forces early validation of the OpenAI API key.
 *
 * Triggers initialization of the shared API key so the application fails fast when OPENAI_API_KEY is missing.
 *
 * @throws IllegalStateException if OPENAI_API_KEY is not set.
 */
private fun validateApiKey() {
    // Force initialization to fail fast if OPENAI_API_KEY is missing.
    openAiApiKey
}

private val openAiApiKey: String by lazy { defaultApiKey() }

private val sharedChatModel: dev.langchain4j.model.openai.OpenAiChatModel by lazy { defaultChatModel() }
private val sharedJson =
    Json {
        prettyPrint = true
        encodeDefaults = true
    }

/**
 * Loads documents from the configured input directory, splits them into chunks, and records the results in the
 * pipeline context.
 *
 * Stores the chunk list under `context.state["chunks"]`, the corresponding text units under
 * `context.state["text_units"]`,
 * and writes a small descriptor file to `context.outputStorage` noting the number of chunks and timestamp.
 *
 * @param config Provides the `inputDir` from which documents are loaded.
 * @param context Pipeline run context where produced artifacts and the descriptor are stored.
 * @return A WorkflowResult whose `result` field is `"documents"`.
 */
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

/**
 * Extracts entities and relationships from the document chunks in the pipeline context.
 *
 * Stores extracted entities under "entities" and relationships under "relationships" in `context.state`,
 * writes a summary entry "graph_extracted.txt" to `context.outputStorage`, and returns a workflow result
 * indicating the graph extraction step.
 *
 * @return A WorkflowResult whose `result` is `"graph"`.
 */
@Suppress("UnusedParameter")
private suspend fun extractGraph(
    config: GraphRagConfig,
    context: PipelineRunContext,
): WorkflowResult {
    val chunks = context.state.getList<DocumentChunk>("chunks")
    val chatModel = sharedChatModel

    val extractor = ExtractGraphWorkflow(chatModel)
    val result =
        extractor.extract(
            chunks = chunks,
            cache = context.cache,
            options = config.extractGraphOptions,
            progress = { context.callbacks.progress(it) },
        )
    context.state["entities"] = result.entities
    context.state["relationships"] = result.relationships
    context.outputStorage.set(
        "graph_extracted.txt",
        "Extracted ${result.entities.size} entities and ${result.relationships.size} relationships at ${Instant.now()}",
    )
    return WorkflowResult(result = "graph")
}

/**
 * Extracts claims from document chunks, stores them in the pipeline context, and records a descriptor.
 *
 * Extracts claims from the "chunks" entry in context.state, saves the resulting claims back into context.state
 * under the "claims" key, and writes a "claims_extracted.txt" descriptor containing the number of claims and a
 * timestamp.
 *
 * @param context PipelineRunContext used to read chunks and persist extracted claims and the output descriptor.
 * @return A WorkflowResult with result set to "claims".
 */
@Suppress("UnusedParameter")
private suspend fun extractClaims(
    config: GraphRagConfig,
    context: PipelineRunContext,
): WorkflowResult {
    val chunks = context.state.getList<DocumentChunk>("chunks")
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

/**
 * Generates embeddings for the pipeline's document chunks and records the results.
 *
 * Embeds the chunks with the configured embedding model, stores the resulting embeddings in
 * `context.state["text_embeddings"]`,
 * and writes a descriptor to `context.outputStorage` containing the number of embeddings and a timestamp.
 *
 * @return A WorkflowResult with result `"text_embeddings"`.
 */
@Suppress("UnusedParameter")
private suspend fun embedText(
    config: GraphRagConfig,
    context: PipelineRunContext,
): WorkflowResult {
    val chunks = context.state.getList<DocumentChunk>("chunks")
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

/**
 * Generates embeddings for the pipeline's entities, stores the embeddings in the run state, reports progress,
 * and writes a brief output descriptor.
 *
 * @param config GraphRagConfig for the current run.
 * @param context PipelineRunContext used to read entities, report progress, and persist outputs.
 * @return A WorkflowResult with result "entity_embeddings".
 */
@Suppress("UnusedParameter")
private suspend fun embedGraph(
    config: GraphRagConfig,
    context: PipelineRunContext,
): WorkflowResult {
    val entities = context.state.getList<Entity>("entities")
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

/**
 * Generates summaries for extracted entities and updates the workflow state and outputs.
 *
 * Summaries produced for entities are stored in the context under "entity_summaries", the
 * relationships in the context are replaced with the (possibly summarized) relationships,
 * and a brief descriptor noting the number of summaries and timestamp is written to outputStorage.
 *
 * @param context PipelineRunContext used to read inputs from and write artifacts to the pipeline state and
 * output storage.
 * @return `WorkflowResult` with `result` set to "entity_summaries".
 */
@Suppress("UnusedParameter")
private suspend fun summarizeDescriptions(
    config: GraphRagConfig,
    context: PipelineRunContext,
): WorkflowResult {
    val entities = context.state.getList<Entity>("entities")
    val relationships = context.state.getList<Relationship>("relationships")
    val chatModel = sharedChatModel
    val summarizer = SummarizeDescriptionsWorkflow(chatModel)
    val textUnits = context.state.getList<TextUnit>("text_units")
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
    val entities = context.state.getList<Entity>("entities")
    val relationships = context.state.getList<Relationship>("relationships")
    val graph = GraphBuilder().buildGraph(entities, relationships)
    context.state["graph"] = graph
    context.outputStorage.set(
        "graph_built.txt",
        "Graph constructed with ${graph.vertexSet().size} nodes at ${Instant.now()}",
    )
    return WorkflowResult(result = "graph_built")
}

/**
 * Writes pipeline artifacts (Parquet files, JSON reports, and a local vector store) to the configured output
 * directory and records an output marker.
 *
 * Serializes community reports as pretty-printed JSON, writes entities, relationships, text embeddings, entity
 * embeddings, community assignments, text units, entity summaries, and claims as Parquet files, saves a
 * LocalVectorStore with text and entity embeddings, and records a timestamped descriptor in the pipeline's
 * output storage.
 *
 * @param config Configuration containing the output directory and other pipeline settings.
 * @param context Pipeline run context holding state and outputStorage used to read artifacts and record the
 * write marker.
 * @return A WorkflowResult with result set to `"outputs_written"`.
 */
@Suppress("UnusedParameter")
private suspend fun writeOutputs(
    config: GraphRagConfig,
    context: PipelineRunContext,
): WorkflowResult {
    val entities = context.state.getList<Entity>("entities")
    val relationships = context.state.getList<Relationship>("relationships")
    val writer = ParquetWriterHelper()
    val textEmbeddings = context.state.getList<TextEmbedding>("text_embeddings")
    val entityEmbeddings = context.state.getList<EntityEmbedding>("entity_embeddings")
    val communities = context.state.getList<CommunityAssignment>("communities")
    val reports = context.state.getList<CommunityReport>("community_reports")
    val claims = context.state.getList<Claim>("claims")
    val textUnits = context.state.getList<TextUnit>("text_units")
    val summaries = context.state.getList<EntitySummary>("entity_summaries")
    val reportsJson =
        sharedJson.encodeToString(
            ListSerializer(CommunityReport.serializer()),
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
    val entities = context.state.getList<Entity>("entities")
    val relationships = context.state.getList<Relationship>("relationships")
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

/**
 * Generate narrative reports for detected communities and store them in the pipeline context.
 *
 * Extracts communities, entities, relationships, text units, claims, and prior reports from the
 * provided PipelineRunContext, runs the CommunityReportWorkflow to produce reports, saves the
 * generated reports back into context.state under "community_reports", and writes a short
 * descriptor to outputStorage.
 *
 * @param config Pipeline configuration used for this workflow step.
 * @param context Mutable pipeline run context holding state and output storage for this run.
 * @return A WorkflowResult whose `result` is "community_reports" after the reports are stored in the context.
 */
@Suppress("UnusedParameter")
private suspend fun communityReports(
    config: GraphRagConfig,
    context: PipelineRunContext,
): WorkflowResult {
    val communities = context.state.getList<CommunityAssignment>("communities")
    val entities = context.state.getList<Entity>("entities")
    val relationships = context.state.getList<Relationship>("relationships")
    val textUnits = context.state.getList<TextUnit>("text_units")
    val claims = context.state.getList<Claim>("claims")
    val hierarchy =
        (context.state["community_hierarchy"] as? Map<*, *> ?: emptyMap<Any, Any>())
            .mapNotNull { (k, v) ->
                val key = k as? Int
                val value = v as? Int
                if (key != null) key to value else null
            }.toMap()
    val priorReports = context.state.getList<CommunityReport>("community_reports")
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

/**
 * Creates a shared OpenAiChatModel configured for the pipeline.
 *
 * The model is configured to use the package's OpenAI API key, the "gpt-4o-mini"
 * model name, JSON Schema response formatting capability, and strict JSON schema validation.
 *
 * @return An OpenAiChatModel configured with the shared API key, model name, supported capability
 *         RESPONSE_FORMAT_JSON_SCHEMA, and strict JSON schema enforcement.
 */
private fun defaultChatModel(): dev.langchain4j.model.openai.OpenAiChatModel =
    dev.langchain4j.model.openai.OpenAiChatModel
        .builder()
        .apiKey(openAiApiKey)
        .modelName("gpt-4o-mini")
        .supportedCapabilities(RESPONSE_FORMAT_JSON_SCHEMA)
        .strictJsonSchema(true)
        .build()

/**
 * Retrieves the OpenAI API key from the OPENAI_API_KEY environment variable and fails fast if missing.
 *
 * @return The OpenAI API key string.
 * @throws IllegalStateException if `OPENAI_API_KEY` is not set.
 */
private fun defaultApiKey(): String =
    System.getenv("OPENAI_API_KEY")
        ?: error("OPENAI_API_KEY environment variable is required")
