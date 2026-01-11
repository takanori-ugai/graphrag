package com.microsoft.graphrag.query

import com.microsoft.graphrag.index.CommunityAssignment
import com.microsoft.graphrag.index.CommunityReport
import com.microsoft.graphrag.index.Entity
import com.microsoft.graphrag.index.EntityEmbedding
import com.microsoft.graphrag.index.EntitySummary
import com.microsoft.graphrag.index.LocalVectorStore
import com.microsoft.graphrag.index.StateCodec
import com.microsoft.graphrag.index.TextEmbedding
import com.microsoft.graphrag.index.TextUnit
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import java.nio.file.Files
import java.nio.file.Path

data class QueryIndexData(
    val textUnits: List<TextUnit>,
    val textEmbeddings: List<TextEmbedding>,
    val entities: List<Entity>,
    val entityEmbeddings: List<EntityEmbedding>,
    val entitySummaries: List<EntitySummary>,
    val communities: List<CommunityAssignment>,
    val communityReports: List<CommunityReport>,
    val communityHierarchy: Map<Int, Int>,
    val vectorStore: LocalVectorStore,
)

/**
 * Load query-time assets (context.json + vector_store.json) produced by the Kotlin indexer.
 */
class QueryIndexLoader(
    private val outputDir: Path,
) {
    fun load(): QueryIndexData {
        val contextPath = outputDir.resolve("context.json")
        require(Files.exists(contextPath)) { "context.json not found in $outputDir. Run the index pipeline first." }
        val rawJson = Files.readString(contextPath)
        val encodedState: Map<String, JsonElement> = Json.decodeFromString(StateCodec.stateSerializer, rawJson)
        val decoded = StateCodec.decodeState(encodedState)

        val textUnits = decoded["text_units"] as? List<TextUnit> ?: emptyList()
        val textEmbeddings = decoded["text_embeddings"] as? List<TextEmbedding> ?: emptyList()
        val entities = decoded["entities"] as? List<Entity> ?: emptyList()
        val entityEmbeddings = decoded["entity_embeddings"] as? List<EntityEmbedding> ?: emptyList()
        val entitySummaries = decoded["entity_summaries"] as? List<EntitySummary> ?: emptyList()
        val communities = decoded["communities"] as? List<CommunityAssignment> ?: emptyList()
        val communityReports = decoded["community_reports"] as? List<CommunityReport> ?: emptyList()
        val communityHierarchy = decoded["community_hierarchy"] as? Map<Int, Int> ?: emptyMap()

        return QueryIndexData(
            textUnits = textUnits,
            textEmbeddings = textEmbeddings,
            entities = entities,
            entityEmbeddings = entityEmbeddings,
            entitySummaries = entitySummaries,
            communities = communities,
            communityReports = communityReports,
            communityHierarchy = communityHierarchy,
            vectorStore = LocalVectorStore(outputDir.resolve("vector_store.json")),
        )
    }
}
