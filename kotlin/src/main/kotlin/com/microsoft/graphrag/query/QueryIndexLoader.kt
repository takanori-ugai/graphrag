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

data class IndexLookup(
    val indexNames: Set<String> = emptySet(),
    val textUnitIndex: Map<String, String> = emptyMap(),
    val entityIndex: Map<String, String> = emptyMap(),
    val communityIndex: Map<Int, String> = emptyMap(),
    val reportIndex: Map<String, String> = emptyMap(),
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
    val indexLookup: IndexLookup = IndexLookup(),
)

class QueryIndexLoader(
    private val outputDirs: List<QueryIndexConfig>,
) {
    constructor(outputDir: Path) : this(listOf(QueryIndexConfig("output", outputDir)))

    private val json = Json { ignoreUnknownKeys = true }

    fun load(): QueryIndexData {
        val resolvedOutputs = resolveOutputs(outputDirs)
        require(resolvedOutputs.isNotEmpty()) {
            "No index outputs found under ${outputDirs.joinToString { it.toString() }}. Run the index pipeline first."
        }

        val partials = resolvedOutputs.map { loadOutput(it) }
        return mergeOutputs(resolvedOutputs, partials)
    }

    private fun resolveOutputs(candidates: List<QueryIndexConfig>): List<QueryIndexConfig> {
        val outputs = mutableListOf<QueryIndexConfig>()
        candidates.forEach { candidate ->
            val normalized = candidate.path.toAbsolutePath().normalize()
            if (looksLikeIndexDir(normalized)) {
                outputs.add(candidate.copy(path = normalized))
            } else if (Files.exists(normalized) && normalized.isDirectory()) {
                Files
                    .walk(normalized, 2)
                    .use { stream ->
                        stream
                            .filter { Files.isDirectory(it) }
                            .filter { looksLikeIndexDir(it) }
                            .forEach { found ->
                                outputs.add(
                                    QueryIndexConfig(
                                        name = "${candidate.name}_${found.fileName}",
                                        path = found,
                                    ),
                                )
                            }
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

    private fun loadOutput(outputDir: QueryIndexConfig): PartialIndex {
        val state = readContextState(outputDir.path)

        val textUnits = loadTextUnits(outputDir.path).ifEmpty { state["text_units"] as? List<TextUnit> ?: emptyList() }
        val textEmbeddings =
            loadTextEmbeddings(outputDir.path).ifEmpty { state["text_embeddings"] as? List<TextEmbedding> ?: emptyList() }
        val entities = loadEntities(outputDir.path).ifEmpty { state["entities"] as? List<Entity> ?: emptyList() }
        val entityEmbeddings =
            loadEntityEmbeddings(outputDir.path).ifEmpty {
                state["entity_embeddings"] as? List<EntityEmbedding> ?: emptyList()
            }
        val entitySummaries =
            loadEntitySummaries(outputDir.path).ifEmpty { state["entity_summaries"] as? List<EntitySummary> ?: emptyList() }
        val relationships =
            loadRelationships(outputDir.path).ifEmpty { state["relationships"] as? List<Relationship> ?: emptyList() }
        val claims = loadClaims(outputDir.path).ifEmpty { state["claims"] as? List<Claim> ?: emptyList() }
        val covariates = loadCovariates(outputDir.path, state)
        val communities =
            loadCommunityAssignments(outputDir.path)
                .ifEmpty { state["communities"] as? List<CommunityAssignment> ?: emptyList() }
        val communityReports =
            loadCommunityReports(outputDir.path)
                .ifEmpty { state["community_reports"] as? List<CommunityReport> ?: emptyList() }
        val communityReportEmbeddings =
            loadCommunityReportEmbeddings(outputDir.path).ifEmpty {
                state["community_report_embeddings"] as? List<CommunityReportEmbedding> ?: emptyList()
            }
        val communityHierarchy =
            loadCommunityHierarchy(outputDir.path)
                ?: (state["community_hierarchy"] as? Map<Int, Int> ?: emptyMap())
        val stats = loadStats(outputDir.path)
        val vectorStorePath = outputDir.path.resolve("vector_store.json")
        val vectorPayload = LocalVectorStore(vectorStorePath).load()

        return PartialIndex(
            name = outputDir.name,
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
        outputs: List<QueryIndexConfig>,
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

        val applyTags = outputs.size > 1
        var communityOffset = 0
        val remapped =
            partials.mapIndexed { idx, partial ->
                val name = outputs.getOrNull(idx)?.name ?: partial.name
                val result = remapPartial(partial, name, communityOffset, applyTags)
                if (applyTags) {
                    communityOffset = result.nextCommunityOffset
                }
                result
            }
        val updatedPartials = remapped.map { it.partial }
        val lookup = mergeLookups(remapped.map { it.lookup })

        val textUnits = mergeUnique(updatedPartials.map { it.textUnits }) { it.id }
        val entities = mergeUnique(updatedPartials.map { it.entities }) { it.id }
        val entitySummaries = mergeUnique(updatedPartials.map { it.entitySummaries }) { it.entityId }
        val relationships =
            mergeUnique(updatedPartials.map { it.relationships }) { rel ->
                rel.id ?: "${rel.sourceId}:${rel.targetId}:${rel.type}"
            }
        val claims = updatedPartials.flatMap { it.claims }
        val communities =
            mergeUnique(updatedPartials.map { it.communities }) { "${it.entityId}:${it.communityId}" }
        val communityReports =
            mergeUnique(updatedPartials.map { it.communityReports }) { report ->
                report.id ?: "${report.communityId}:${report.parentCommunityId ?: -1}"
            }
        val communityReportEmbeddings =
            mergeUnique(updatedPartials.map { it.communityReportEmbeddings }) { it.communityId }
        val communityHierarchy = linkedMapOf<Int, Int>()
        updatedPartials.forEach { part -> part.communityHierarchy.forEach { (k, v) -> communityHierarchy[k] = v } }
        val covariates = mergeCovariates(updatedPartials)
        val stats = updatedPartials.mapNotNull { it.stats }

        val textEmbeddings =
            mergeUnique(updatedPartials.map { it.textEmbeddings }) { it.chunkId }.ifEmpty {
                updatedPartials.mapNotNull { it.vectorPayload?.textEmbeddings }.flatten()
            }
        val entityEmbeddings =
            mergeUnique(updatedPartials.map { it.entityEmbeddings }) { it.entityId }.ifEmpty {
                updatedPartials.mapNotNull { it.vectorPayload?.entityEmbeddings }.flatten()
            }

        val mergedPayload =
            Payload(
                textEmbeddings = textEmbeddings,
                entityEmbeddings = entityEmbeddings,
            )
        val vectorStorePath =
            updatedPartials.firstNotNullOfOrNull { it.vectorStorePath }
                ?: outputs.first().path.resolve("vector_store.json")

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
            indexLookup = lookup,
        )
    }

    private data class RemapResult(
        val partial: PartialIndex,
        val lookup: IndexLookup,
        val nextCommunityOffset: Int,
    )

    private fun remapPartial(
        partial: PartialIndex,
        name: String,
        communityOffset: Int,
        applyTags: Boolean,
    ): RemapResult {
        val safeName = sanitizeName(name)
        if (!applyTags) {
            val lookup =
                IndexLookup(
                    indexNames = setOf(safeName),
                    textUnitIndex = partial.textUnits.associate { it.id to safeName },
                    entityIndex =
                        partial.entities.associate { it.id to safeName } +
                            partial.entities.mapNotNull { entity ->
                                entity.shortId?.let { it to safeName }
                            },
                    communityIndex = buildCommunityIndex(partial, safeName),
                    reportIndex = buildReportIndex(partial, safeName),
                )
            return RemapResult(partial.copy(name = safeName), lookup, communityOffset)
        }

        val prefix = "$safeName-"
        val communityShift = mutableMapOf<Int, Int>()
        val shift: (Int?) -> Int? = { id ->
            id?.let { value ->
                if (value < 0) {
                    value
                } else {
                    communityShift.getOrPut(value) { value + communityOffset }
                }
            }
        }

        val textUnits = partial.textUnits.map { unit -> unit.copy(id = prefix + unit.id, chunkId = prefix + unit.chunkId) }
        val textEmbeddings = partial.textEmbeddings.map { embedding -> embedding.copy(chunkId = prefix + embedding.chunkId) }
        val entities =
            partial.entities.map { entity ->
                entity.copy(
                    id = prefix + entity.id,
                    sourceChunkId = prefix + entity.sourceChunkId,
                    communityIds = entity.communityIds.mapNotNull { shift(it) },
                    textUnitIds = entity.textUnitIds.map { prefix + it },
                    shortId = entity.shortId?.let { prefix + it } ?: prefix + entity.id,
                )
            }
        val entityEmbeddings = partial.entityEmbeddings.map { it.copy(entityId = prefix + it.entityId) }
        val entitySummaries = partial.entitySummaries.map { it.copy(entityId = prefix + it.entityId) }
        val relationships =
            partial.relationships.map { rel ->
                rel.copy(
                    sourceId = prefix + rel.sourceId,
                    targetId = prefix + rel.targetId,
                    sourceChunkId = prefix + rel.sourceChunkId,
                    textUnitIds = rel.textUnitIds.map { prefix + it },
                    id = rel.id?.let { prefix + it },
                )
            }
        val claims = partial.claims
        val covariates =
            partial.covariates.mapValues { (_, values) ->
                values.map { cov ->
                    cov.copy(
                        id = prefix + cov.id.ifBlank { "${cov.subjectId}-${cov.covariateType}" },
                        subjectId = prefix + cov.subjectId,
                    )
                }
            }
        val communities =
            partial.communities.map { assignment ->
                val shifted = shift(assignment.communityId) ?: assignment.communityId + communityOffset
                assignment.copy(
                    entityId = prefix + assignment.entityId,
                    communityId = shifted,
                )
            }
        val communityReports =
            partial.communityReports.map { report ->
                val shiftedId = shift(report.communityId) ?: (report.communityId + communityOffset)
                val shiftedParent = shift(report.parentCommunityId)
                report.copy(
                    communityId = shiftedId,
                    parentCommunityId = shiftedParent,
                    id = report.id?.let { "$prefix$it" },
                    shortId = report.shortId?.let { "$prefix$it" } ?: "$prefix$shiftedId",
                    attributes = report.attributes + ("index_name" to safeName),
                )
            }
        val communityReportEmbeddings =
            partial.communityReportEmbeddings.map { embedding ->
                embedding.copy(communityId = shift(embedding.communityId) ?: (embedding.communityId + communityOffset))
            }
        val communityHierarchy =
            partial.communityHierarchy
                .mapKeys { shift(it.key) ?: (it.key + communityOffset) }
                .mapValues { shift(it.value) ?: (it.value + communityOffset) }

        val shiftedPartial =
            PartialIndex(
                name = safeName,
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
                vectorStorePath = partial.vectorStorePath,
                vectorPayload = rewriteVectorPayload(partial.vectorPayload, prefix),
                stats = partial.stats,
            )
        val maxCommunity = communityShift.values.maxOrNull() ?: (communityOffset - 1)
        val nextOffset = if (maxCommunity < communityOffset) communityOffset else maxCommunity + 1
        val lookup =
            IndexLookup(
                indexNames = setOf(safeName),
                textUnitIndex = textUnits.associate { it.id to safeName },
                entityIndex =
                    entities.associate { it.id to safeName } +
                        entities.mapNotNull { entity ->
                            entity.shortId?.let { it to safeName }
                        },
                communityIndex = buildCommunityIndex(shiftedPartial, safeName),
                reportIndex = buildReportIndex(shiftedPartial, safeName),
            )
        return RemapResult(shiftedPartial, lookup, nextOffset)
    }

    private fun mergeLookups(lookups: List<IndexLookup>): IndexLookup =
        IndexLookup(
            indexNames = lookups.flatMap { it.indexNames }.toSet(),
            textUnitIndex = lookups.flatMap { it.textUnitIndex.entries }.associate { it.toPair() },
            entityIndex = lookups.flatMap { it.entityIndex.entries }.associate { it.toPair() },
            communityIndex = lookups.flatMap { it.communityIndex.entries }.associate { it.toPair() },
            reportIndex = lookups.flatMap { it.reportIndex.entries }.associate { it.toPair() },
        )

    private fun buildCommunityIndex(
        partial: PartialIndex,
        name: String,
    ): Map<Int, String> {
        val ids = mutableSetOf<Int>()
        ids.addAll(partial.communityReports.map { it.communityId })
        ids.addAll(partial.communities.map { it.communityId })
        ids.addAll(partial.communityHierarchy.keys.filter { it >= 0 })
        ids.addAll(partial.communityHierarchy.values.filter { it >= 0 })
        return ids.associateWith { name }
    }

    private fun buildReportIndex(
        partial: PartialIndex,
        name: String,
    ): Map<String, String> {
        val mapping = mutableMapOf<String, String>()
        partial.communityReports.forEach { report ->
            report.id?.let { mapping[it] = name }
            report.shortId?.let { mapping[it] = name }
            mapping[report.communityId.toString()] = name
        }
        return mapping
    }

    private fun rewriteVectorPayload(
        payload: Payload?,
        prefix: String,
    ): Payload? {
        payload ?: return null
        val textEmbeddings = payload.textEmbeddings.map { it.copy(chunkId = prefix + it.chunkId) }
        val entityEmbeddings = payload.entityEmbeddings.map { it.copy(entityId = prefix + it.entityId) }
        return Payload(textEmbeddings = textEmbeddings, entityEmbeddings = entityEmbeddings)
    }

    private fun sanitizeName(name: String): String = name.trim().ifBlank { "index" }.replace("\\s+".toRegex(), "_")

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
    val name: String,
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
