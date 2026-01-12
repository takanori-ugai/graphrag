package com.microsoft.graphrag.query

import com.microsoft.graphrag.index.Claim
import com.microsoft.graphrag.index.CommunityAssignment
import com.microsoft.graphrag.index.CommunityReport
import com.microsoft.graphrag.index.CommunityReportEmbedding
import com.microsoft.graphrag.index.Covariate
import com.microsoft.graphrag.index.Entity
import com.microsoft.graphrag.index.EntityEmbedding
import com.microsoft.graphrag.index.EntitySummary
import com.microsoft.graphrag.index.LocalVectorStore
import com.microsoft.graphrag.index.Payload
import com.microsoft.graphrag.index.Relationship
import com.microsoft.graphrag.index.StateCodec
import com.microsoft.graphrag.index.TextEmbedding
import com.microsoft.graphrag.index.TextUnit
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.MapSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import org.apache.parquet.example.data.Group
import org.apache.parquet.hadoop.ParquetReader
import org.apache.parquet.hadoop.example.GroupReadSupport
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.isDirectory
import org.apache.hadoop.fs.Path as HadoopPath

@Serializable
data class IndexStats(
    val workflows: Map<String, Map<String, Double>> = emptyMap(),
    val totalRuntime: Double = 0.0,
)

data class QueryIndexData(
    val textUnits: List<TextUnit>,
    val textEmbeddings: List<TextEmbedding>,
    val entities: List<Entity>,
    val entityEmbeddings: List<EntityEmbedding>,
    val entitySummaries: List<EntitySummary>,
    val relationships: List<Relationship>,
    val claims: List<Claim>,
    val covariates: Map<String, List<Covariate>>,
    val communities: List<CommunityAssignment>,
    val communityReports: List<CommunityReport>,
    val communityReportEmbeddings: List<CommunityReportEmbedding>,
    val communityHierarchy: Map<Int, Int>,
    val stats: List<IndexStats> = emptyList(),
    val vectorStore: LocalVectorStore,
)

/**
 * Load query-time assets from parquet/JSON outputs. Supports multiple output directories
 * (e.g., base + updates) and falls back to context.json when parquet files are missing.
 */
class QueryIndexLoader(
    private val outputDirs: List<Path>,
) {
    constructor(outputDir: Path) : this(listOf(outputDir))

    private val json = Json { ignoreUnknownKeys = true }

    fun load(): QueryIndexData {
        val resolvedOutputs = resolveOutputs(outputDirs)
        require(resolvedOutputs.isNotEmpty()) {
            "No index outputs found under ${outputDirs.joinToString { it.toString() }}. Run the index pipeline first."
        }

        val partials = resolvedOutputs.map { loadOutput(it) }
        return mergeOutputs(resolvedOutputs, partials)
    }

    private fun resolveOutputs(candidates: List<Path>): List<Path> {
        val outputs = mutableListOf<Path>()
        candidates.forEach { candidate ->
            val normalized = candidate.toAbsolutePath().normalize()
            if (looksLikeIndexDir(normalized)) {
                outputs.add(normalized)
            } else if (Files.exists(normalized) && normalized.isDirectory()) {
                Files
                    .walk(normalized, 2)
                    .use { stream ->
                        stream
                            .filter { Files.isDirectory(it) }
                            .filter { looksLikeIndexDir(it) }
                            .forEach { outputs.add(it) }
                    }
            }
        }
        return outputs.distinct()
    }

    private fun looksLikeIndexDir(dir: Path): Boolean {
        if (!Files.isDirectory(dir)) return false
        val markers =
            listOf(
                "text_units.parquet",
                "entities.parquet",
                "context.json",
                "vector_store.json",
            )
        return markers.any { Files.exists(dir.resolve(it)) }
    }

    private fun loadOutput(outputDir: Path): PartialIndex {
        val state = readContextState(outputDir)

        val textUnits = loadTextUnits(outputDir).ifEmpty { state["text_units"] as? List<TextUnit> ?: emptyList() }
        val textEmbeddings =
            loadTextEmbeddings(outputDir).ifEmpty { state["text_embeddings"] as? List<TextEmbedding> ?: emptyList() }
        val entities = loadEntities(outputDir).ifEmpty { state["entities"] as? List<Entity> ?: emptyList() }
        val entityEmbeddings =
            loadEntityEmbeddings(outputDir).ifEmpty { state["entity_embeddings"] as? List<EntityEmbedding> ?: emptyList() }
        val entitySummaries =
            loadEntitySummaries(outputDir).ifEmpty { state["entity_summaries"] as? List<EntitySummary> ?: emptyList() }
        val relationships =
            loadRelationships(outputDir).ifEmpty { state["relationships"] as? List<Relationship> ?: emptyList() }
        val claims = loadClaims(outputDir).ifEmpty { state["claims"] as? List<Claim> ?: emptyList() }
        val covariates = loadCovariates(outputDir, state)
        val communities =
            loadCommunityAssignments(outputDir).ifEmpty { state["communities"] as? List<CommunityAssignment> ?: emptyList() }
        val communityReports =
            loadCommunityReports(outputDir).ifEmpty { state["community_reports"] as? List<CommunityReport> ?: emptyList() }
        val communityReportEmbeddings =
            loadCommunityReportEmbeddings(outputDir).ifEmpty {
                state["community_report_embeddings"] as? List<CommunityReportEmbedding> ?: emptyList()
            }
        val communityHierarchy =
            loadCommunityHierarchy(outputDir)
                ?: (state["community_hierarchy"] as? Map<Int, Int> ?: emptyMap())
        val stats = loadStats(outputDir)
        val vectorStorePath = outputDir.resolve("vector_store.json")
        val vectorPayload = LocalVectorStore(vectorStorePath).load()

        return PartialIndex(
            textUnits = textUnits,
            textEmbeddings = textEmbeddings,
            entities = entities,
            entityEmbeddings = entityEmbeddings,
            entitySummaries = entitySummaries,
            relationships = relationships,
            claims = claims,
            covariates = covariates,
            communities = communities,
            communityReports = communityReports,
            communityReportEmbeddings = communityReportEmbeddings,
            communityHierarchy = communityHierarchy,
            vectorStorePath = vectorStorePath,
            vectorPayload = vectorPayload,
            stats = stats,
        )
    }

    private fun mergeOutputs(
        outputs: List<Path>,
        partials: List<PartialIndex>,
    ): QueryIndexData {
        fun <T, K> mergeUnique(
            values: List<List<T>>,
            key: (T) -> K,
        ): List<T> {
            val acc = linkedMapOf<K, T>()
            values.flatten().forEach { acc[key(it)] = it }
            return acc.values.toList()
        }

        val textUnits = mergeUnique(partials.map { it.textUnits }) { it.id }
        val entities = mergeUnique(partials.map { it.entities }) { it.id }
        val entitySummaries = mergeUnique(partials.map { it.entitySummaries }) { it.entityId }
        val relationships =
            mergeUnique(partials.map { it.relationships }) { rel ->
                rel.id ?: "${rel.sourceId}:${rel.targetId}:${rel.type}"
            }
        val claims = partials.flatMap { it.claims }
        val communities =
            mergeUnique(partials.map { it.communities }) { "${it.entityId}:${it.communityId}" }
        val communityReports =
            mergeUnique(partials.map { it.communityReports }) { report ->
                report.id ?: "${report.communityId}:${report.parentCommunityId ?: -1}"
            }
        val communityReportEmbeddings =
            mergeUnique(partials.map { it.communityReportEmbeddings }) { it.communityId }
        val communityHierarchy = linkedMapOf<Int, Int>()
        partials.forEach { part -> part.communityHierarchy.forEach { (k, v) -> communityHierarchy[k] = v } }
        val covariates = mergeCovariates(partials)
        val stats = partials.mapNotNull { it.stats }

        val textEmbeddings =
            mergeUnique(partials.map { it.textEmbeddings }) { it.chunkId }.ifEmpty {
                partials.mapNotNull { it.vectorPayload?.textEmbeddings }.flatten()
            }
        val entityEmbeddings =
            mergeUnique(partials.map { it.entityEmbeddings }) { it.entityId }.ifEmpty {
                partials.mapNotNull { it.vectorPayload?.entityEmbeddings }.flatten()
            }

        val mergedPayload =
            Payload(
                textEmbeddings = textEmbeddings,
                entityEmbeddings = entityEmbeddings,
            )
        val vectorStorePath =
            partials.firstNotNullOfOrNull { it.vectorStorePath }
                ?: outputs.first().resolve("vector_store.json")

        return QueryIndexData(
            textUnits = textUnits,
            textEmbeddings = textEmbeddings,
            entities = entities,
            entityEmbeddings = entityEmbeddings,
            entitySummaries = entitySummaries,
            relationships = relationships,
            claims = claims,
            covariates = covariates,
            communities = communities,
            communityReports = communityReports,
            communityReportEmbeddings = communityReportEmbeddings,
            communityHierarchy = communityHierarchy,
            stats = stats,
            vectorStore = LocalVectorStore(vectorStorePath, mergedPayload),
        )
    }

    private fun mergeCovariates(partials: List<PartialIndex>): Map<String, List<Covariate>> {
        val grouped = linkedMapOf<String, MutableMap<String, Covariate>>()
        partials.forEach { partial ->
            partial.covariates.forEach { (type, values) ->
                val existing = grouped.getOrPut(type) { linkedMapOf() }
                values.forEach { cov ->
                    val key = cov.id.ifBlank { "${cov.subjectId}:${cov.attributes.hashCode()}" }
                    val mergedAttributes = existing[key]?.attributes.orEmpty() + cov.attributes
                    existing[key] = cov.copy(attributes = mergedAttributes)
                }
            }
        }
        return grouped.mapValues { (_, value) -> value.values.toList() }
    }

    private fun readContextState(outputDir: Path): Map<String, Any?> {
        val contextPath = outputDir.resolve("context.json")
        if (!Files.exists(contextPath)) return emptyMap()
        return runCatching {
            val rawJson = Files.readString(contextPath)
            val encodedState: Map<String, JsonElement> = Json.decodeFromString(StateCodec.stateSerializer, rawJson)
            StateCodec.decodeState(encodedState)
        }.getOrDefault(emptyMap())
    }

    private fun loadStats(outputDir: Path): IndexStats? {
        val statsPath = outputDir.resolve("stats.json")
        if (!Files.exists(statsPath)) return null
        return runCatching { json.decodeFromString(IndexStats.serializer(), Files.readString(statsPath)) }.getOrNull()
    }

    private fun loadCommunityHierarchy(outputDir: Path): Map<Int, Int>? {
        val jsonPath = outputDir.resolve("community_hierarchy.json")
        if (!Files.exists(jsonPath)) return null
        return runCatching {
            json.decodeFromString(MapSerializer(Int.serializer(), Int.serializer()), Files.readString(jsonPath))
        }.getOrNull()
    }

    private fun loadTextUnits(outputDir: Path): List<TextUnit> = loadTextUnitsFromParquet(outputDir.resolve("text_units.parquet"))

    private fun loadTextEmbeddings(outputDir: Path): List<TextEmbedding> =
        loadTextEmbeddingsFromParquet(outputDir.resolve("text_embeddings.parquet"))

    private fun loadEntities(outputDir: Path): List<Entity> = loadEntitiesFromParquet(outputDir.resolve("entities.parquet"))

    private fun loadEntityEmbeddings(outputDir: Path): List<EntityEmbedding> =
        loadEntityEmbeddingsFromParquet(outputDir.resolve("entity_embeddings.parquet"))

    private fun loadEntitySummaries(outputDir: Path): List<EntitySummary> =
        loadEntitySummariesFromParquet(outputDir.resolve("entity_summaries.parquet"))

    private fun loadRelationships(outputDir: Path): List<Relationship> =
        loadRelationshipsFromParquet(outputDir.resolve("relationships.parquet"))

    private fun loadClaims(outputDir: Path): List<Claim> = loadClaimsFromParquet(outputDir.resolve("claims.parquet"))

    private fun loadCommunityAssignments(outputDir: Path): List<CommunityAssignment> =
        loadCommunitiesFromParquet(outputDir.resolve("communities.parquet"))

    private fun loadCommunityReportEmbeddings(outputDir: Path): List<CommunityReportEmbedding> =
        loadCommunityReportEmbeddingsFromParquet(outputDir.resolve("community_report_embeddings.parquet"))

    private fun loadCommunityReports(outputDir: Path): List<CommunityReport> {
        val jsonPath = outputDir.resolve("community_reports.json")
        if (!Files.exists(jsonPath)) return emptyList()
        return runCatching {
            json.decodeFromString(
                kotlinx.serialization.builtins.ListSerializer(CommunityReport.serializer()),
                Files.readString(jsonPath),
            )
        }.getOrDefault(emptyList())
    }

    private fun loadCovariates(
        outputDir: Path,
        state: Map<String, Any?>,
    ): Map<String, List<Covariate>> {
        val parquet = loadCovariatesFromParquet(outputDir.resolve("covariates.parquet"))
        if (parquet.isNotEmpty()) {
            return parquet.groupBy { it.covariateType }
        }
        @Suppress("UNCHECKED_CAST")
        return state["covariates"] as? Map<String, List<Covariate>> ?: emptyMap()
    }

    private fun loadTextUnitsFromParquet(path: Path): List<TextUnit> =
        readParquet(path) { group ->
            val id = group.string("id") ?: return@readParquet null
            val chunkId = group.string("chunk_id", "chunkId") ?: return@readParquet null
            val text = group.string("text") ?: ""
            val sourcePath = group.string("source", "source_path", "sourcePath") ?: ""
            TextUnit(id, chunkId, text, sourcePath)
        }

    private fun loadEntitiesFromParquet(path: Path): List<Entity> =
        readParquet(path) { group ->
            val id = group.string("id", "entity_id") ?: return@readParquet null
            val name = group.string("name", "display_name") ?: id
            val type = group.string("type", "category") ?: ""
            val sourceChunkId = group.string("source_chunk_id", "sourceChunkId", "source_id", "chunk_id") ?: ""
            val description = group.string("description")
            val rank = group.double("rank")
            val shortId = group.string("short_id", "shortId")
            val communityIds = group.intList("community_ids")
            val textUnitIds = group.stringList("text_unit_ids", "textUnitIds")
            val attributes = parseAttributes(group.string("attributes", "properties"))
            Entity(
                id = id,
                name = name,
                type = type,
                sourceChunkId = sourceChunkId,
                description = description,
                rank = rank,
                shortId = shortId,
                communityIds = communityIds,
                textUnitIds = textUnitIds,
                attributes = attributes,
            )
        }

    private fun loadRelationshipsFromParquet(path: Path): List<Relationship> =
        readParquet(path) { group ->
            val sourceId = group.string("source_id", "sourceId", "source") ?: return@readParquet null
            val targetId = group.string("target_id", "targetId", "target") ?: return@readParquet null
            val type = group.string("type", "relationship", "predicate") ?: ""
            val description = group.string("description", "text")
            val sourceChunkId = group.string("source_chunk_id", "sourceChunkId", "chunk_id", "chunkId") ?: ""
            val weight = group.double("weight", "relationship_weight")
            val rank = group.double("rank")
            val attributes = parseAttributes(group.string("attributes", "properties"))
            val shortId = group.string("short_id", "shortId")
            val id = group.string("id")
            val textUnitIds = group.stringList("text_unit_ids", "textUnitIds")
            Relationship(
                sourceId = sourceId,
                targetId = targetId,
                type = type,
                description = description,
                sourceChunkId = sourceChunkId,
                weight = weight,
                rank = rank,
                attributes = attributes,
                shortId = shortId,
                id = id,
                textUnitIds = textUnitIds,
            )
        }

    private fun loadClaimsFromParquet(path: Path): List<Claim> =
        readParquet(path) { group ->
            val subject = group.string("subject", "source") ?: return@readParquet null
            val obj = group.string("object", "target") ?: return@readParquet null
            val claimType = group.string("claimType", "claim_type", "type") ?: ""
            val status = group.string("status") ?: ""
            val startDate = group.string("startDate", "start_date") ?: ""
            val endDate = group.string("endDate", "end_date") ?: ""
            val description = group.string("description", "text") ?: ""
            val sourceText = group.string("sourceText", "source_text") ?: ""
            Claim(
                subject = subject,
                `object` = obj,
                claimType = claimType,
                status = status,
                startDate = startDate,
                endDate = endDate,
                description = description,
                sourceText = sourceText,
            )
        }

    private fun loadTextEmbeddingsFromParquet(path: Path): List<TextEmbedding> =
        readParquet(path) { group ->
            val chunkId = group.string("chunk_id", "chunkId", "id") ?: return@readParquet null
            val vector = group.doubleList("vector", "embedding", "values")
            TextEmbedding(chunkId, vector)
        }

    private fun loadEntityEmbeddingsFromParquet(path: Path): List<EntityEmbedding> =
        readParquet(path) { group ->
            val entityId = group.string("entity_id", "entityId", "id") ?: return@readParquet null
            val vector = group.doubleList("vector", "embedding", "values")
            EntityEmbedding(entityId, vector)
        }

    private fun loadCommunityReportEmbeddingsFromParquet(path: Path): List<CommunityReportEmbedding> =
        readParquet(path) { group ->
            val communityId = group.int("community_id", "communityId") ?: return@readParquet null
            val vector = group.doubleList("vector", "embedding", "values")
            CommunityReportEmbedding(communityId, vector)
        }

    private fun loadCommunitiesFromParquet(path: Path): List<CommunityAssignment> =
        readParquet(path) { group ->
            val entityId = group.string("entity_id", "entityId", "id") ?: return@readParquet null
            val communityId = group.int("community_id", "communityId", "community") ?: return@readParquet null
            CommunityAssignment(entityId, communityId)
        }

    private fun loadEntitySummariesFromParquet(path: Path): List<EntitySummary> =
        readParquet(path) { group ->
            val entityId = group.string("entity_id", "entityId", "id") ?: return@readParquet null
            val summary = group.string("summary", "description") ?: return@readParquet null
            EntitySummary(entityId, summary)
        }

    private fun loadCovariatesFromParquet(path: Path): List<Covariate> =
        readParquet(path) { group ->
            val id = group.string("id") ?: ""
            val subjectId = group.string("subject_id", "entity_id", "subjectId") ?: return@readParquet null
            val covariateType = group.string("covariate_type", "type") ?: "covariate"
            val attrKey = group.string("name", "key")
            val attrValue = group.string("value")
            val attributes =
                when {
                    attrKey != null && attrValue != null -> mapOf(attrKey to attrValue)
                    else -> parseAttributes(group.string("attributes"))
                }
            Covariate(
                id = id.ifBlank { "$subjectId-$covariateType" },
                subjectId = subjectId,
                covariateType = covariateType,
                attributes = attributes,
            )
        }

    private fun parseAttributes(raw: String?): Map<String, String> {
        if (raw.isNullOrBlank()) return emptyMap()
        return runCatching {
            val element = json.parseToJsonElement(raw)
            if (element is JsonObject) {
                element.mapValues { (_, v) ->
                    when (v) {
                        is kotlinx.serialization.json.JsonPrimitive -> v.contentOrNull ?: v.toString()
                        else -> v.toString()
                    }
                }
            } else {
                emptyMap()
            }
        }.getOrElse { emptyMap() }
    }

    private fun <T> readParquet(
        path: Path,
        parser: (Group) -> T?,
    ): List<T> {
        if (!Files.exists(path)) return emptyList()
        return runCatching {
            val hadoopPath = HadoopPath(path.toAbsolutePath().normalize().toString())
            ParquetReader.builder(GroupReadSupport(), hadoopPath).build().use { reader ->
                generateSequence { reader.read() }
                    .mapNotNull { runCatching { parser(it) }.getOrNull() }
                    .toList()
            }
        }.getOrDefault(emptyList())
    }

    private fun Group.string(vararg candidates: String): String? =
        candidates.firstNotNullOfOrNull { name ->
            if (!hasField(name)) return@firstNotNullOfOrNull null
            val count = getFieldRepetitionCount(name)
            if (count == 0) return@firstNotNullOfOrNull null
            getString(name, 0)
        }

    private fun Group.double(vararg candidates: String): Double? =
        candidates.firstNotNullOfOrNull { name ->
            if (!hasField(name)) return@firstNotNullOfOrNull null
            if (getFieldRepetitionCount(name) == 0) return@firstNotNullOfOrNull null
            getDouble(name, 0)
        }

    private fun Group.int(vararg candidates: String): Int? =
        candidates.firstNotNullOfOrNull { name ->
            if (!hasField(name)) return@firstNotNullOfOrNull null
            if (getFieldRepetitionCount(name) == 0) return@firstNotNullOfOrNull null
            getInteger(name, 0)
        }

    private fun Group.intList(vararg candidates: String): List<Int> =
        candidates
            .asSequence()
            .mapNotNull { name ->
                if (!hasField(name)) return@mapNotNull null
                (0 until getFieldRepetitionCount(name)).map { idx -> getInteger(name, idx) }
            }.firstOrNull() ?: emptyList()

    private fun Group.doubleList(vararg candidates: String): List<Double> =
        candidates
            .asSequence()
            .mapNotNull { name ->
                if (!hasField(name)) return@mapNotNull null
                (0 until getFieldRepetitionCount(name)).map { idx -> getDouble(name, idx) }
            }.firstOrNull() ?: emptyList()

    private fun Group.stringList(vararg candidates: String): List<String> =
        candidates
            .asSequence()
            .mapNotNull { name ->
                if (!hasField(name)) return@mapNotNull null
                (0 until getFieldRepetitionCount(name)).map { idx -> getString(name, idx) }
            }.firstOrNull() ?: emptyList()

    private fun Group.hasField(name: String): Boolean =
        try {
            type.getFieldIndex(name)
            true
        } catch (_: Exception) {
            false
        }
}

private data class PartialIndex(
    val textUnits: List<TextUnit>,
    val textEmbeddings: List<TextEmbedding>,
    val entities: List<Entity>,
    val entityEmbeddings: List<EntityEmbedding>,
    val entitySummaries: List<EntitySummary>,
    val relationships: List<Relationship>,
    val claims: List<Claim>,
    val covariates: Map<String, List<Covariate>>,
    val communities: List<CommunityAssignment>,
    val communityReports: List<CommunityReport>,
    val communityReportEmbeddings: List<CommunityReportEmbedding>,
    val communityHierarchy: Map<Int, Int>,
    val vectorStorePath: Path,
    val vectorPayload: Payload?,
    val stats: IndexStats?,
)
