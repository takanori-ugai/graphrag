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
import io.github.oshai.kotlinlogging.KotlinLogging
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
    private val logger = KotlinLogging.logger {}

    /**
     * Load and merge configured query index outputs into a single QueryIndexData.
     *
     * Resolves the configured output directories, loads each partial index, and merges them into a unified index.
     *
     * @return The merged QueryIndexData containing all loaded assets and a combined LocalVectorStore.
     * @throws IllegalArgumentException If no index outputs are found for the configured output directories.
     */
    fun load(): QueryIndexData {
        val resolvedOutputs = resolveOutputs(outputDirs)
        require(resolvedOutputs.isNotEmpty()) {
            "No index outputs found under ${outputDirs.joinToString { it.toString() }}. Run the index pipeline first."
        }

        val partials = resolvedOutputs.map { loadOutput(it) }
        return mergeOutputs(resolvedOutputs, partials)
    }

    /**
     * Locate concrete index output directories from a list of candidate paths.
     *
     * For each candidate path, if it directly appears to be an index directory it is returned
     * (with a normalized absolute path). If the candidate is a directory but does not itself
     * look like an index, its subtree (up to depth 2) is searched for subdirectories that do.
     * Subdirectories found are returned as new QueryIndexConfig entries whose names are composed
     * from the original candidate name and the found directory name.
     *
     * @param candidates Candidate index configurations to resolve into concrete index directories.
     * @return A list of distinct `QueryIndexConfig` objects pointing to resolved index directories.
     */
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

    /**
     * Determines whether the given path looks like a query index directory by checking for known marker files.
     *
     * @param dir Filesystem path to inspect; must point to a directory to be considered.
     * @return `true` if `dir` is a directory and contains at least one of the expected marker files
     * (`text_units.parquet`, `entities.parquet`, `context.json`, or `vector_store.json`), `false` otherwise.
     */
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

    /**
     * Load all index components from a single output directory and assemble them into a PartialIndex.
     *
     * Loads text units, embeddings, entities, summaries, relationships, claims, covariates,
     * community assignments, community reports and embeddings, community hierarchy, stats,
     * and the vector store payload for the provided output. Missing on-disk artifacts are
     * populated from the output's context state when available.
     *
     * @param outputDir Configuration object that provides the output directory path and logical name.
     * @return A PartialIndex containing the loaded components, the resolved vector store path, and the loaded
     * vector payload.
     */
    private fun loadOutput(outputDir: QueryIndexConfig): PartialIndex {
        val state = readContextState(outputDir.path)

        val textUnits = loadTextUnits(outputDir.path).ifEmpty { stateList<TextUnit>(state, "text_units") }
        val textEmbeddings =
            loadTextEmbeddings(outputDir.path).ifEmpty { stateList<TextEmbedding>(state, "text_embeddings") }
        val entities = loadEntities(outputDir.path).ifEmpty { stateList<Entity>(state, "entities") }
        val entityEmbeddings = loadEntityEmbeddings(outputDir.path).ifEmpty { stateList<EntityEmbedding>(state, "entity_embeddings") }
        val entitySummaries = loadEntitySummaries(outputDir.path).ifEmpty { stateList<EntitySummary>(state, "entity_summaries") }
        val relationships = loadRelationships(outputDir.path).ifEmpty { stateList<Relationship>(state, "relationships") }
        val claims = loadClaims(outputDir.path).ifEmpty { stateList<Claim>(state, "claims") }
        val covariates = loadCovariates(outputDir.path, state)
        val communities =
            loadCommunityAssignments(outputDir.path)
                .ifEmpty { stateList<CommunityAssignment>(state, "communities") }
        val communityReports =
            loadCommunityReports(outputDir.path)
                .ifEmpty { stateList<CommunityReport>(state, "community_reports") }
        val communityReportEmbeddings =
            loadCommunityReportEmbeddings(outputDir.path).ifEmpty {
                stateList<CommunityReportEmbedding>(state, "community_report_embeddings")
            }
        val communityHierarchy =
            loadCommunityHierarchy(outputDir.path) ?: stateIntMap(state, "community_hierarchy")
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

    /**
     * Merge multiple loaded partial index outputs into a single unified QueryIndexData for query-time use.
     *
     * Merges and deduplicates index components from the provided partials, optionally applies per-output namespacing
     * when more than one output is present, consolidates community id space, composes a combined vector payload and
     * vector store path, and builds a unified index lookup mapping.
     *
     * @param outputs The original list of QueryIndexConfig entries corresponding to the loaded partials; used to
     *   derive per-output names and a fallback vector store path.
     * @param partials The PartialIndex instances loaded from each output directory to be merged.
     * @return A QueryIndexData containing merged and deduplicated text units, embeddings, entities, summaries,
     *   relationships, claims, covariates, communities, community reports and embeddings, community hierarchy,
     *   accumulated stats, a LocalVectorStore with the merged payload and path, and a combined IndexLookup.
     */
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

    /**
     * Remaps identifiers in a PartialIndex to a namespaced form when needed and produces an IndexLookup for the
     * remapped data.
     *
     * When `applyTags` is false this returns the same partial with its name sanitized and an IndexLookup that
     * maps the partial's existing IDs to that sanitized name.
     * When `applyTags` is true this returns a new PartialIndex whose IDs are prefixed with the sanitized index
     * name and whose community IDs are shifted to avoid collisions using `communityOffset`; the vector payload
     * is similarly rewritten to reference prefixed IDs.
     *
     * @param partial The PartialIndex to remap.
     * @param name The index name to sanitize and (when tagging) use as a prefix.
     * @param communityOffset The starting offset used to shift non-negative community IDs to avoid collisions
     * across outputs.
     * @param applyTags If true, prefix and shift IDs; if false, keep IDs unchanged but record the sanitized
     * name in the lookup.
     * @return A RemapResult containing the (possibly remapped) PartialIndex, an IndexLookup mapping remapped
     * IDs to the sanitized index name, and the next community offset to use for subsequent remaps.
     */
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

        val textUnits = prefixTextUnits(partial.textUnits, prefix)
        val textEmbeddings = prefixTextEmbeddings(partial.textEmbeddings, prefix)
        val entities = prefixEntities(partial.entities, prefix, shift)
        val entityEmbeddings = prefixEntityEmbeddings(partial.entityEmbeddings, prefix)
        val entitySummaries = prefixEntitySummaries(partial.entitySummaries, prefix)
        val relationships = prefixRelationships(partial.relationships, prefix)
        val claims = partial.claims
        val covariates = prefixCovariates(partial.covariates, prefix)
        val communities = prefixCommunities(partial.communities, prefix, shift, communityOffset)
        val communityReports = prefixCommunityReports(partial.communityReports, prefix, shift, communityOffset, safeName)
        val communityReportEmbeddings = prefixCommunityReportEmbeddings(partial.communityReportEmbeddings, shift, communityOffset)
        val communityHierarchy = shiftCommunityHierarchy(partial.communityHierarchy, shift, communityOffset)

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

    /**
     * Merge multiple IndexLookup instances into a single combined lookup.
     *
     * @param lookups The list of IndexLookup objects to merge.
     * @return An IndexLookup whose `indexNames` is the union of all input `indexNames`, and whose
     * `textUnitIndex`, `entityIndex`, `communityIndex`, and `reportIndex` contain entries from all inputs; when
     * the same key appears in multiple inputs, the entry from the later element in `lookups` takes precedence.
     */
    private fun mergeLookups(lookups: List<IndexLookup>): IndexLookup =
        IndexLookup(
            indexNames = lookups.flatMap { it.indexNames }.toSet(),
            textUnitIndex = lookups.flatMap { it.textUnitIndex.entries }.associate { it.toPair() },
            entityIndex = lookups.flatMap { it.entityIndex.entries }.associate { it.toPair() },
            communityIndex = lookups.flatMap { it.communityIndex.entries }.associate { it.toPair() },
            reportIndex = lookups.flatMap { it.reportIndex.entries }.associate { it.toPair() },
        )

    /**
     * Collects community-related IDs from a partial index and maps each to the given index name.
     *
     * @param partial The PartialIndex to extract community IDs from (communityReports, communities, and
     * communityHierarchy).
     * @param name The index name to associate with each collected community ID.
     * @return A map from each discovered community ID to `name`.
     */
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

    /**
     * Builds a mapping from community report identifiers to the provided index name.
     *
     * The mapping includes each report's `id`, `shortId` (if present), and the report's
     * `communityId` converted to a string, all mapped to `name`.
     *
     * @param partial PartialIndex containing the communityReports to index.
     * @param name The index name to associate with each report identifier.
     * @return A map from report identifier (id, shortId, or communityId as string) to `name`.
     */
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

    /**
     * Prefix embedding identifiers in a payload with the given string.
     *
     * @param payload The payload containing text and entity embeddings; may be null.
     * @param prefix The string to prepend to each embedding's chunk or entity identifier.
     * @return A new Payload with each text embedding's `chunkId` and each entity embedding's `entityId`
     * prefixed by `prefix`, or `null` if `payload` is null.
     */
    private fun rewriteVectorPayload(
        payload: Payload?,
        prefix: String,
    ): Payload? {
        payload ?: return null
        val textEmbeddings = payload.textEmbeddings.map { it.copy(chunkId = prefix + it.chunkId) }
        val entityEmbeddings = payload.entityEmbeddings.map { it.copy(entityId = prefix + it.entityId) }
        return Payload(textEmbeddings = textEmbeddings, entityEmbeddings = entityEmbeddings)
    }

    /**
     * Prefixes each TextUnit's `id` and `chunkId` with the given prefix.
     *
     * @param units The list of TextUnit objects to transform.
     * @param prefix The string to prepend to each unit's `id` and `chunkId`.
     * @return A new list of TextUnit where every `id` and `chunkId` is prefixed by `prefix`.
     */
    private fun prefixTextUnits(
        units: List<TextUnit>,
        prefix: String,
    ): List<TextUnit> = units.map { unit -> unit.copy(id = prefix + unit.id, chunkId = prefix + unit.chunkId) }

    /**
     * Prefixes the chunkId of each text embedding with the provided string.
     *
     * @param prefix String to prepend to each embedding's `chunkId`.
     * @return A list of `TextEmbedding` where each element has its `chunkId` set to `prefix + originalChunkId`.
     */
    private fun prefixTextEmbeddings(
        embeddings: List<TextEmbedding>,
        prefix: String,
    ): List<TextEmbedding> = embeddings.map { embedding -> embedding.copy(chunkId = prefix + embedding.chunkId) }

    /**
     * Prefixes entity identifiers and remaps their community IDs.
     *
     * Applies `prefix` to each entity's `id`, `sourceChunkId`, `textUnitIds`, and `shortId` (falls back to the
     * prefixed original id when `shortId` is null). Applies `shiftCommunity` to each community id and omits any
     * that map to `null`.
     *
     * @param entities The list of entities to transform.
     * @param prefix String to prepend to identifier fields.
     * @param shiftCommunity Function that maps an original community id to a new id or `null` to drop that association.
     * @return A list of entities with prefixed identifiers and remapped `communityIds` (with `null` results removed).
     */
    private fun prefixEntities(
        entities: List<Entity>,
        prefix: String,
        shiftCommunity: (Int?) -> Int?,
    ): List<Entity> =
        entities.map { entity ->
            entity.copy(
                id = prefix + entity.id,
                sourceChunkId = prefix + entity.sourceChunkId,
                communityIds = entity.communityIds.mapNotNull { shiftCommunity(it) },
                textUnitIds = entity.textUnitIds.map { prefix + it },
                shortId = entity.shortId?.let { prefix + it } ?: prefix + entity.id,
            )
        }

    /**
     * Prefixes each embedding's `entityId` with the given string.
     *
     * @param embeddings The list of EntityEmbedding objects to transform.
     * @param prefix The string to prepend to each embedding's `entityId`.
     * @return A new list of EntityEmbedding with `entityId` values prefixed by `prefix`.
     */
    private fun prefixEntityEmbeddings(
        embeddings: List<EntityEmbedding>,
        prefix: String,
    ): List<EntityEmbedding> = embeddings.map { it.copy(entityId = prefix + it.entityId) }

    /**
     * Prefixes the `entityId` of each EntitySummary with the given prefix.
     *
     * @param summaries The EntitySummary objects to transform.
     * @param prefix The string to prepend to each EntitySummary's `entityId`.
     * @return A list of EntitySummary where each `entityId` is prefixed.
     */
    private fun prefixEntitySummaries(
        summaries: List<EntitySummary>,
        prefix: String,
    ): List<EntitySummary> = summaries.map { it.copy(entityId = prefix + it.entityId) }

    /**
     * Prefixes all identifier fields on each Relationship with the given string.
     *
     * @param relationships The relationships whose id fields and referenced chunk/text-unit ids will be prefixed.
     * @param prefix The string to prepend to each identifier.
     * @return A new list of Relationship where `sourceId`, `targetId`, `sourceChunkId`, each entry in
     * `textUnitIds`, and `id` (if present) are prefixed.
     */
    private fun prefixRelationships(
        relationships: List<Relationship>,
        prefix: String,
    ): List<Relationship> =
        relationships.map { rel ->
            rel.copy(
                sourceId = prefix + rel.sourceId,
                targetId = prefix + rel.targetId,
                sourceChunkId = prefix + rel.sourceChunkId,
                textUnitIds = rel.textUnitIds.map { prefix + it },
                id = rel.id?.let { prefix + it },
            )
        }

    /**
     * Prefixes each covariate's `id` and `subjectId` with the provided namespace and ensures every covariate
     * has an `id`.
     *
     * For covariates whose `id` is blank, an id is generated as "<subjectId>-<covariateType>" before applying
     * the prefix.
     *
     * @param covariates Map from covariate type to list of covariates to be prefixed.
     * @param prefix String to prepend to each covariate `id` and `subjectId`.
     * @return A new map with the same keys where each `Covariate` has its `id` and `subjectId` prefixed.
     */
    private fun prefixCovariates(
        covariates: Map<String, List<Covariate>>,
        prefix: String,
    ): Map<String, List<Covariate>> =
        covariates.mapValues { (_, values) ->
            values.map { cov ->
                cov.copy(
                    id = prefix + cov.id.ifBlank { "${cov.subjectId}-${cov.covariateType}" },
                    subjectId = prefix + cov.subjectId,
                )
            }
        }

    /**
     * Prefixes each assignment's entity ID and adjusts its community ID using the provided shift or offset.
     *
     * @param communities The list of community assignments to transform.
     * @param prefix String to prepend to each assignment's `entityId`.
     * @param shift Function that maps an original community ID to a new one; if it returns `null`, the result
     * falls back to `originalCommunityId + communityOffset`.
     * @param communityOffset Integer offset added to the original community ID when `shift` returns `null`.
     * @return A new list of `CommunityAssignment` with prefixed `entityId` values and updated `communityId` values.
     */
    private fun prefixCommunities(
        communities: List<CommunityAssignment>,
        prefix: String,
        shift: (Int?) -> Int?,
        communityOffset: Int,
    ): List<CommunityAssignment> =
        communities.map { assignment ->
            val shifted = shift(assignment.communityId) ?: assignment.communityId + communityOffset
            assignment.copy(
                entityId = prefix + assignment.entityId,
                communityId = shifted,
            )
        }

    /**
     * Produce a new list of CommunityReport with identifiers namespaced by `prefix`, community IDs adjusted via
     * `shift` (or offset), and an added `index_name` attribute.
     *
     * For each report:
     * - `communityId` is replaced by the result of `shift(report.communityId)` if non-null, otherwise
     * `report.communityId + communityOffset`.
     * - `parentCommunityId` is replaced by `shift(report.parentCommunityId)` (may be null).
     * - `id` and `shortId` are prefixed with `prefix`; if `shortId` is missing, it falls back to the prefixed
     * shifted community id.
     * - `attributes` is augmented with the key `"index_name"` set to `indexName`.
     *
     * @param reports The original community reports to transform.
     * @param prefix String to prepend to report `id` and `shortId`.
     * @param shift Function that maps an original community id to a new id or returns null to indicate the
     * fallback offset should be used.
     * @param communityOffset Integer offset added to the original community id when `shift` returns null.
     * @param indexName Value stored in the `"index_name"` attribute for each transformed report.
     * @return A list of transformed CommunityReport objects with namespaced ids, shifted community ids, and
     * updated attributes.
     */
    private fun prefixCommunityReports(
        reports: List<CommunityReport>,
        prefix: String,
        shift: (Int?) -> Int?,
        communityOffset: Int,
        indexName: String,
    ): List<CommunityReport> =
        reports.map { report ->
            val shiftedId = shift(report.communityId) ?: (report.communityId + communityOffset)
            val shiftedParent = shift(report.parentCommunityId)
            report.copy(
                communityId = shiftedId,
                parentCommunityId = shiftedParent,
                id = report.id?.let { "$prefix$it" },
                shortId = report.shortId?.let { "$prefix$it" } ?: "$prefix$shiftedId",
                attributes = report.attributes + ("index_name" to indexName),
            )
        }

    /**
     * Update community IDs on a list of community report embeddings by applying a remapping or offset.
     *
     * @param embeddings The list of CommunityReportEmbedding objects to update.
     * @param shift A function that takes an original community ID (nullable) and returns a remapped community
     * ID or `null` to indicate fallback.
     * @param communityOffset Integer offset to add to the original community ID when `shift` returns `null`.
     * @return A new list of CommunityReportEmbedding with `communityId` values replaced by the remapped or offset IDs.
     */
    private fun prefixCommunityReportEmbeddings(
        embeddings: List<CommunityReportEmbedding>,
        shift: (Int?) -> Int?,
        communityOffset: Int,
    ): List<CommunityReportEmbedding> =
        embeddings.map { embedding ->
            embedding.copy(communityId = shift(embedding.communityId) ?: (embedding.communityId + communityOffset))
        }

    /**
     * Applies an ID transformation to every community and parent entry in a community hierarchy.
     *
     * For each key and value, calls `shift(id)` and uses the returned value if non-null;
     * otherwise falls back to `id + communityOffset`.
     *
     * @param hierarchy Map from community id to parent community id.
     * @param shift Function that maps an original community id to a new id, or returns `null` to request the
     * offset fallback.
     * @param communityOffset Integer offset added to an id when `shift` returns `null`.
     * @return A new map where both community ids and parent ids have been transformed according to `shift` or
     * offset fallback.
     */
    private fun shiftCommunityHierarchy(
        hierarchy: Map<Int, Int>,
        shift: (Int?) -> Int?,
        communityOffset: Int,
    ): Map<Int, Int> =
        hierarchy.mapKeys { shift(it.key) ?: (it.key + communityOffset) }.mapValues { shift(it.value) ?: (it.value + communityOffset) }

    /**
     * Consolidates covariates from multiple partial indices into a map keyed by covariate type.
     *
     * For each covariate type, covariates with the same id (or, if id is blank, the same
     * subjectId and attributes hash) are deduplicated and their attribute maps are merged;
     * attributes from later covariates override earlier ones for the same key.
     *
     * @param partials The list of partial indices whose covariates will be merged.
     * @return A map from covariate type to the list of merged `Covariate` instances for that type.
     */
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

    /**
     * Reads and decodes the context.json file from the given output directory into a state map.
     *
     * If the file is missing or decoding fails, an empty map is returned.
     *
     * @return A map of state keys to decoded values, or an empty map if the file is missing or cannot be decoded.
     */
    private fun readContextState(outputDir: Path): Map<String, Any?> {
        val contextPath = outputDir.resolve("context.json")
        if (!Files.exists(contextPath)) return emptyMap()
        return runCatching {
            val rawJson = Files.readString(contextPath)
            val encodedState: Map<String, JsonElement> = Json.decodeFromString(StateCodec.stateSerializer, rawJson)
            StateCodec.decodeState(encodedState)
        }.getOrDefault(emptyMap())
    }

    /**
     * Load IndexStats from a stats.json file located in the given output directory.
     *
     * @return `IndexStats` parsed from stats.json, or `null` if the file is missing or cannot be decoded.
     */
    private fun loadStats(outputDir: Path): IndexStats? {
        val statsPath = outputDir.resolve("stats.json")
        if (!Files.exists(statsPath)) return null
        return runCatching { json.decodeFromString(IndexStats.serializer(), Files.readString(statsPath)) }.getOrNull()
    }

    /**
     * Retrieve a typed list from a state map by key.
     *
     * Returns the elements at `state[key]` filtered to type `T`. If the key is missing or the value is not a
     * list, an empty list is returned. If the list contains mixed types, only the entries of type `T` are
     * returned and a warning is logged.
     *
     * @param state Map containing arbitrary state values.
     * @param key Key whose associated value should be interpreted as a list of `T`.
     * @return A list of `T` extracted from the state entry, or an empty list if not present or not a list.
     */
    private inline fun <reified T> stateList(
        state: Map<String, Any?>,
        key: String,
    ): List<T> {
        val value = state[key] ?: return emptyList()
        if (value !is List<*>) {
            logger.warn { "State key '$key' expected List<${T::class.simpleName}> but was ${value::class.simpleName}" }
            return emptyList()
        }
        val typed = value.filterIsInstance<T>()
        if (typed.size != value.size) {
            logger.warn { "State key '$key' contains mixed types; kept ${typed.size} of ${value.size} entries" }
        }
        return typed
    }

    /**
     * Extracts a Map<Int, Int> from a heterogeneous state map for the specified key.
     *
     * If the key is missing or the value is not a map, an empty map is returned and a warning is logged.
     * Entries whose keys or values are not numeric are ignored; a warning is logged if any entries are dropped.
     *
     * @param state A map of string keys to arbitrary state values.
     * @param key The key whose value should be interpreted as a map of integers.
     * @return A map of integer keys to integer values parsed from the state, or an empty map on missing/invalid data.
     */
    private fun stateIntMap(
        state: Map<String, Any?>,
        key: String,
    ): Map<Int, Int> {
        val value = state[key] ?: return emptyMap()
        val map =
            value as? Map<*, *> ?: run {
                logger.warn { "State key '$key' expected Map<Int, Int> but was ${value::class.simpleName}" }
                return emptyMap()
            }
        val typed =
            map.mapNotNull { (k, v) ->
                val keyInt = (k as? Number)?.toInt() ?: return@mapNotNull null
                val valInt = (v as? Number)?.toInt() ?: return@mapNotNull null
                keyInt to valInt
            }
        if (typed.size != map.size) {
            logger.warn { "State key '$key' contains non-integer entries; kept ${typed.size} of ${map.size} pairs" }
        }
        return typed.toMap()
    }

    /**
     * Loads the community hierarchy from `community_hierarchy.json` in the given output directory.
     *
     * @return A map from community id to parent community id if the file exists and is valid, `null` otherwise.
     */
    private fun loadCommunityHierarchy(outputDir: Path): Map<Int, Int>? {
        val jsonPath = outputDir.resolve("community_hierarchy.json")
        if (!Files.exists(jsonPath)) return null
        return runCatching {
            json.decodeFromString(MapSerializer(Int.serializer(), Int.serializer()), Files.readString(jsonPath))
        }.getOrNull()
    }

    /**
     * Loads text units from the index output directory's Parquet file.
     *
     * @param outputDir The index output directory containing `text_units.parquet`.
     * @return A list of `TextUnit` objects read from `text_units.parquet`; returns an empty list if the file is
     * missing or cannot be parsed.
     */
    private fun loadTextUnits(outputDir: Path): List<TextUnit> = loadTextUnitsFromParquet(outputDir.resolve("text_units.parquet"))

    /**
     * Load text embeddings from the output directory's `text_embeddings.parquet` file.
     *
     * @param outputDir Path to the index output directory containing `text_embeddings.parquet`.
     * @return A list of `TextEmbedding` objects parsed from `text_embeddings.parquet`; returns an empty list if
     * the file is missing or cannot be read.
     */
    private fun loadTextEmbeddings(outputDir: Path): List<TextEmbedding> =
        loadTextEmbeddingsFromParquet(outputDir.resolve("text_embeddings.parquet"))

    /**
     * Loads entity records from the given index output directory.
     *
     * Reads the `entities.parquet` file inside `outputDir` and parses it into a list of `Entity`.
     *
     * @param outputDir Path to the index output directory containing `entities.parquet`.
     * @return A list of parsed `Entity` objects; an empty list if the file is missing or cannot be parsed.
     */
    private fun loadEntities(outputDir: Path): List<Entity> = loadEntitiesFromParquet(outputDir.resolve("entities.parquet"))

    /**
     * Load entity embeddings from the specified index output directory.
     *
     * @param outputDir The path to the index output directory to read embeddings from.
     * @return A list of loaded EntityEmbedding objects, or an empty list if no embeddings are available.
     */
    private fun loadEntityEmbeddings(outputDir: Path): List<EntityEmbedding> =
        loadEntityEmbeddingsFromParquet(outputDir.resolve("entity_embeddings.parquet"))

    /**
     * Load entity summary records from the given index output directory.
     *
     * @param outputDir Path to the index output directory that may contain `entity_summaries.parquet`.
     * @return A list of `EntitySummary` objects read from `entity_summaries.parquet`, or an empty list if the
     * file is missing or cannot be parsed.
     */
    private fun loadEntitySummaries(outputDir: Path): List<EntitySummary> =
        loadEntitySummariesFromParquet(outputDir.resolve("entity_summaries.parquet"))

    /**
     * Load relationship records from the output directory's relationships.parquet file.
     *
     * @return A list of Relationship objects parsed from `relationships.parquet`; returns an empty list if the
     * file is missing or cannot be read/parsed.
     */
    private fun loadRelationships(outputDir: Path): List<Relationship> =
        loadRelationshipsFromParquet(outputDir.resolve("relationships.parquet"))

    /**
     * Load claim records from the output directory's claims.parquet file.
     *
     * @return A list of `Claim` objects parsed from `claims.parquet`, or an empty list if the file is missing
     * or cannot be read.
     */
    private fun loadClaims(outputDir: Path): List<Claim> = loadClaimsFromParquet(outputDir.resolve("claims.parquet"))

    /**
     * Loads community assignment records from the output directory's communities.parquet file.
     *
     * @param outputDir The path to the index output directory that should contain `communities.parquet`.
     * @return A list of loaded CommunityAssignment objects, or an empty list if none are found.
     */
    private fun loadCommunityAssignments(outputDir: Path): List<CommunityAssignment> =
        loadCommunitiesFromParquet(outputDir.resolve("communities.parquet"))

    /**
     * Loads community report embedding records from the given index output directory.
     *
     * @param outputDir Path to the index output directory that should contain `community_report_embeddings.parquet`.
     * @return A list of `CommunityReportEmbedding` parsed from `community_report_embeddings.parquet`, or an
     * empty list if the file is missing or cannot be read.
     */
    private fun loadCommunityReportEmbeddings(outputDir: Path): List<CommunityReportEmbedding> =
        loadCommunityReportEmbeddingsFromParquet(outputDir.resolve("community_report_embeddings.parquet"))

    /**
     * Loads community reports from a "community_reports.json" file inside the given output directory.
     *
     * @return A list of CommunityReport parsed from the file; returns an empty list if the file is missing or
     * parsing fails.
     */
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

    /**
     * Load covariates for an index, preferring a parquet file and falling back to the provided state.
     *
     * Attempts to read `covariates.parquet` in the given output directory and, if present, groups
     * the records by their `covariateType`. If no parquet file is found or it is empty, returns the
     * `covariates` entry from the provided state map or an empty map if absent.
     *
     * @param outputDir Directory containing index output files (used to locate `covariates.parquet`).
     * @param state Context/state map (e.g., decoded from `context.json`) used as a fallback source.
     * @return A map from covariate type to the list of `Covariate` objects; empty if none found.
     */
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

    /**
     * Loads TextUnit records from a Parquet file at the given path into a list of TextUnit.
     *
     * Each record must contain `id` and `chunk_id` (or `chunkId`) to be included. Missing `text`
     * or `source`/`source_path` fields are treated as empty strings.
     *
     * @param path Path to the Parquet file to read.
     * @return A list of parsed TextUnit instances.
     */
    private fun loadTextUnitsFromParquet(path: Path): List<TextUnit> =
        readParquet(path) { group ->
            val id = group.string("id") ?: return@readParquet null
            val chunkId = group.string("chunk_id", "chunkId") ?: return@readParquet null
            val text = group.string("text") ?: ""
            val sourcePath = group.string("source", "source_path", "sourcePath") ?: ""
            TextUnit(id, chunkId, text, sourcePath)
        }

    /**
     * Parse entities from the specified Parquet file into a list of Entity instances.
     *
     * @param path The filesystem path to the Parquet file containing entity records.
     * @return A list of parsed Entity objects.
     */
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

    /**
     * Load relationships from a Parquet file, parsing known field variants and attributes.
     *
     * Rows missing a required source or target identifier are skipped. Field names are resolved
     * from multiple common variants (e.g., `source_id` / `sourceId` / `source`, `target_id` / `targetId` / `target`).
     * The `attributes` field is parsed as JSON into a map when present.
     *
     * @param path Path to the Parquet file containing relationship records.
     * @return A list of parsed Relationship objects; rows with missing required identifiers are omitted.
     */
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

    /**
     * Load Claim records from a Parquet file.
     *
     * Parses each row into a Claim, mapping multiple possible field names for common columns.
     * Rows missing a required `subject` or `object` value are skipped. Optional fields default to an empty
     * string when absent.
     *
     * Recognized column name aliases:
     * - subject: "subject" or "source"
     * - object: "object" or "target"
     * - claimType: "claimType", "claim_type", or "type"
     * - status: "status"
     * - startDate: "startDate" or "start_date"
     * - endDate: "endDate" or "end_date"
     * - description: "description" or "text"
     * - sourceText: "sourceText" or "source_text"
     *
     * @param path Path to the Parquet file containing claim rows.
     * @return A list of parsed Claim objects; rows missing required identifiers are omitted.
     */
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

    /**
     * Load text embeddings from a Parquet file into a list of TextEmbedding objects.
     *
     * @param path The path to the Parquet file containing text embedding records.
     * @return A list of parsed TextEmbedding instances; an empty list if no valid records are found.
     */
    private fun loadTextEmbeddingsFromParquet(path: Path): List<TextEmbedding> =
        readParquet(path) { group ->
            val chunkId = group.string("chunk_id", "chunkId", "id") ?: return@readParquet null
            val vector = group.doubleList("vector", "embedding", "values")
            TextEmbedding(chunkId, vector)
        }

    /**
     * Load entity embeddings from a Parquet file into a list of EntityEmbedding.
     *
     * @param path Path to the Parquet file containing entity embedding records.
     * @return A list of EntityEmbedding parsed from the file; rows missing an entity id are skipped.
     */
    private fun loadEntityEmbeddingsFromParquet(path: Path): List<EntityEmbedding> =
        readParquet(path) { group ->
            val entityId = group.string("entity_id", "entityId", "id") ?: return@readParquet null
            val vector = group.doubleList("vector", "embedding", "values")
            EntityEmbedding(entityId, vector)
        }

    /**
     * Load community report embeddings from a Parquet file.
     *
     * Each record's community_id and embedding vector are parsed; records missing a community_id are ignored.
     *
     * @param path Path to the Parquet file containing community report embeddings.
     * @return A list of CommunityReportEmbedding parsed from the file.
     */
    private fun loadCommunityReportEmbeddingsFromParquet(path: Path): List<CommunityReportEmbedding> =
        readParquet(path) { group ->
            val communityId = group.int("community_id", "communityId") ?: return@readParquet null
            val vector = group.doubleList("vector", "embedding", "values")
            CommunityReportEmbedding(communityId, vector)
        }

    /**
     * Load community assignments from a Parquet file.
     *
     * Parses rows that contain `entity_id` (string) and `community_id` (int) and converts them to
     * CommunityAssignment objects.
     * Rows missing either required field are skipped.
     *
     * @param path Path to the Parquet file containing community assignment records.
     * @return A list of parsed CommunityAssignment objects; entries with missing `entity_id` or `community_id`
     * are omitted.
     */
    private fun loadCommunitiesFromParquet(path: Path): List<CommunityAssignment> =
        readParquet(path) { group ->
            val entityId = group.string("entity_id", "entityId", "id") ?: return@readParquet null
            val communityId = group.int("community_id", "communityId", "community") ?: return@readParquet null
            CommunityAssignment(entityId, communityId)
        }

    /**
     * Loads entity summaries from a Parquet file into a list of EntitySummary.
     *
     * Rows that do not contain both an entity id and a summary are skipped.
     *
     * @param path Path to the Parquet file containing entity summary records.
     * @return A list of parsed EntitySummary objects.
     */
    private fun loadEntitySummariesFromParquet(path: Path): List<EntitySummary> =
        readParquet(path) { group ->
            val entityId = group.string("entity_id", "entityId", "id") ?: return@readParquet null
            val summary = group.string("summary", "description") ?: return@readParquet null
            EntitySummary(entityId, summary)
        }

    /**
     * Load covariates from a Parquet file and convert them into a list of Covariate objects.
     *
     * Each record must include a subject id (one of "subject_id", "entity_id", or "subjectId"); records missing
     * a subject id are skipped.
     * If an explicit `id` is blank, an id is synthesized as "<subjectId>-<covariateType>".
     * Attribute key/value pairs are taken from `name`/`value` when present, otherwise parsed from an
     * `attributes` JSON field.
     *
     * @param path Path to the Parquet file containing covariate records.
     * @return A list of parsed Covariate objects; records lacking a subject id are omitted.
     */
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

    /**
     * Parse a JSON-encoded attributes string into a map of attribute names to string values.
     *
     * Returns an empty map if `raw` is null or blank, if the JSON does not represent an object,
     * or if parsing fails.
     *
     * @param raw A JSON string representing an object of attributes (nullable).
     * @return A map where each key is an attribute name and each value is the attribute's string representation.
     */
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

    /**
     * Reads a Parquet file and parses each record into a list of values using the provided parser.
     *
     * If the file does not exist, cannot be read, or if individual record parsing fails, an empty list
     * or the subset of successfully parsed records is returned; failures are logged but do not throw.
     *
     * @param path Filesystem path to the Parquet file.
     * @param parser Function that converts a Parquet `Group` record to an instance of `T`, or `null` to skip
     * the record.
     * @return A list of parsed `T` instances (possibly empty).
     */
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
        }.getOrElse { error ->
            logger.warn { "Failed to read Parquet file at $path ($error)" }
            emptyList()
        }
    }

    /**
     * Retrieve the first available string value from the provided candidate field names.
     *
     * Attempts the candidate names in order and returns the first string value found for a field
     * that exists and contains at least one repetition; returns `null` if no such field/value is found.
     *
     * @param candidates Field names to try, in order of preference.
     * @return The first found string value for a present field with at least one entry, or `null` if none found.
     */
    private fun Group.string(vararg candidates: String): String? =
        candidates.firstNotNullOfOrNull { name ->
            if (!hasField(name)) return@firstNotNullOfOrNull null
            val count = getFieldRepetitionCount(name)
            if (count == 0) return@firstNotNullOfOrNull null
            getString(name, 0)
        }

    /**
     * Retrieve the first available double value from the group using a list of candidate field names.
     *
     * @param candidates Field names to check in order; the first present field with at least one value is used.
     * @return The `Double` value from the first matching field, or `null` if no candidate field is present or
     * contains a value.
     */
    private fun Group.double(vararg candidates: String): Double? =
        candidates.firstNotNullOfOrNull { name ->
            if (!hasField(name)) return@firstNotNullOfOrNull null
            if (getFieldRepetitionCount(name) == 0) return@firstNotNullOfOrNull null
            getDouble(name, 0)
        }

    /**
     * Retrieves the first present integer field from the group using the given candidate names.
     *
     * @param candidates Field names to try in order.
     * @return The first integer value found for the provided field names, or `null` if none are present or have
     * a value.
     */
    private fun Group.int(vararg candidates: String): Int? =
        candidates.firstNotNullOfOrNull { name ->
            if (!hasField(name)) return@firstNotNullOfOrNull null
            if (getFieldRepetitionCount(name) == 0) return@firstNotNullOfOrNull null
            getInteger(name, 0)
        }

    /**
     * Extracts the repeated integer values for the first available field from a list of candidate field names.
     *
     * Checks each candidate name in order and returns the sequence of integers stored in the first field that exists.
     *
     * @param candidates Candidate field names to try, in priority order.
     * @return The list of integers from the first present candidate field, or an empty list if none are present.
     */
    private fun Group.intList(vararg candidates: String): List<Int> =
        candidates
            .asSequence()
            .mapNotNull { name ->
                if (!hasField(name)) return@mapNotNull null
                (0 until getFieldRepetitionCount(name)).map { idx -> getInteger(name, idx) }
            }.firstOrNull() ?: emptyList()

    /**
     * Extracts the repeated double values for the first field name that exists on the Group.
     *
     * @param candidates One or more candidate field names to check; the function uses the first candidate that
     * is present and returns all its repeated double values.
     * @return A list of doubles read from the first present candidate field, or an empty list if none of the
     * candidates exist.
     */
    private fun Group.doubleList(vararg candidates: String): List<Double> =
        candidates
            .asSequence()
            .mapNotNull { name ->
                if (!hasField(name)) return@mapNotNull null
                (0 until getFieldRepetitionCount(name)).map { idx -> getDouble(name, idx) }
            }.firstOrNull() ?: emptyList()

    /**
     * Retrieves the sequence of string values from the first present repeated field among the given candidate names.
     *
     * If multiple candidate field names are provided, the function returns the list of strings for the first
     * candidate that exists in the Group. If none of the candidates exist, an empty list is returned.
     *
     * @param candidates Field names to probe in order of preference.
     * @return A list of strings from the first found repeated field, or an empty list if none are present.
     */
    private fun Group.stringList(vararg candidates: String): List<String> =
        candidates
            .asSequence()
            .mapNotNull { name ->
                if (!hasField(name)) return@mapNotNull null
                (0 until getFieldRepetitionCount(name)).map { idx -> getString(name, idx) }
            }.firstOrNull() ?: emptyList()

    /**
     * Checks whether this Parquet `Group` contains a field with the given name.
     *
     * @param name The field name to check for.
     * @return `true` if the field is present, `false` otherwise.
     */
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
