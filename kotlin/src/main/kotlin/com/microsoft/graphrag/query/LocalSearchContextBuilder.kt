package com.microsoft.graphrag.query

import com.microsoft.graphrag.index.Claim
import com.microsoft.graphrag.index.CommunityAssignment
import com.microsoft.graphrag.index.CommunityReport
import com.microsoft.graphrag.index.Covariate
import com.microsoft.graphrag.index.Entity
import com.microsoft.graphrag.index.EntitySummary
import com.microsoft.graphrag.index.LocalVectorStore
import com.microsoft.graphrag.index.Relationship
import com.microsoft.graphrag.index.TextEmbedding
import com.microsoft.graphrag.index.TextUnit
import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingType
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.output.Response
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Builds local-search context similar to the Python LocalSearchMixedContext: maps queries to entities,
 * assembles entities/relationships/claims/communities/text-unit tables under a token budget, and
 * optionally adds conversation history.
 */
@Suppress("LongParameterList", "TooManyFunctions")
class LocalSearchContextBuilder(
    private val embeddingModel: EmbeddingModel,
    private val vectorStore: LocalVectorStore,
    private val textUnits: List<TextUnit>,
    private val textEmbeddings: List<TextEmbedding>,
    private val entities: List<Entity>,
    private val entitySummaries: List<EntitySummary>,
    private val relationships: List<Relationship>,
    private val claims: List<Claim>,
    private val covariates: Map<String, List<Covariate>>,
    private val communities: List<CommunityAssignment>,
    private val communityReports: List<CommunityReport>,
    private val columnDelimiter: String = "|",
) {
    data class ConversationTurn(
        val role: Role,
        val content: String,
    ) {
        enum class Role {
            USER,
            ASSISTANT,
            SYSTEM,
        }
    }

    data class ConversationHistory(
        val turns: List<ConversationTurn>,
    ) {
        fun getUserTurns(maxTurns: Int): List<String> =
            turns
                .filter { it.role == ConversationTurn.Role.USER }
                .takeLast(maxTurns)
                .map { it.content }
    }

    data class LocalContextResult(
        val contextText: String,
        val contextChunks: List<QueryContextChunk>,
    )

    @Suppress("LongMethod", "ReturnCount", "LongParameterList", "CyclomaticComplexMethod")
    suspend fun buildContext(
        query: String,
        conversationHistory: ConversationHistory? = null,
        includeEntityNames: List<String> = emptyList(),
        excludeEntityNames: List<String> = emptyList(),
        conversationHistoryMaxTurns: Int = 5,
        conversationHistoryUserTurnsOnly: Boolean = true,
        maxContextTokens: Int = 8000,
        textUnitProp: Double = 0.5,
        communityProp: Double = 0.25,
        topKMappedEntities: Int = 10,
        topKRelationships: Int = 10,
        topKClaims: Int = 10,
        topKCommunities: Int = 5,
        includeCommunityRank: Boolean = false,
        includeEntityRank: Boolean = false,
        includeRelationshipWeight: Boolean = false,
        relationshipRankingAttribute: String = "rank",
        useCommunitySummary: Boolean = false,
        minCommunityRank: Int = 0,
    ): LocalContextResult {
        require(communityProp + textUnitProp <= 1.0) { "community_prop + text_unit_prop must be <= 1" }

        val queryForEntities =
            if (conversationHistory != null && conversationHistory.turns.isNotEmpty()) {
                val userTurns =
                    if (conversationHistoryUserTurnsOnly) {
                        conversationHistory.getUserTurns(conversationHistoryMaxTurns)
                    } else {
                        conversationHistory.turns.takeLast(conversationHistoryMaxTurns).map { it.content }
                    }
                (listOf(query) + userTurns).joinToString("\n")
            } else {
                query
            }

        val selectedEntities =
            mapQueryToEntities(
                queryForEntities,
                includeEntityNames,
                excludeEntityNames,
                topKMappedEntities,
                oversample = 2,
            )

        val sections = mutableListOf<String>()
        val usedChunks = mutableListOf<QueryContextChunk>()
        var remainingTokens = maxContextTokens

        // conversation history context
        if (conversationHistory != null && conversationHistory.turns.isNotEmpty()) {
            val historySection =
                buildConversationHistorySection(
                    conversationHistory = conversationHistory,
                    includeUserOnly = conversationHistoryUserTurnsOnly,
                    maxQaTurns = conversationHistoryMaxTurns,
                )
            val tokens = tokenCount(historySection)
            if (tokens <= remainingTokens) {
                sections += historySection
                remainingTokens -= tokens
            }
        }

        // community context
        val communityTokens = (remainingTokens * communityProp).toInt()
        val (communitySection, communitySectionTokens) =
            buildCommunitySection(
                selectedEntities,
                tokenBudget = communityTokens,
                useCommunitySummary = useCommunitySummary,
                includeRank = includeCommunityRank,
                minCommunityRank = minCommunityRank,
                topK = topKCommunities,
            )
        if (communitySection.isNotBlank()) {
            sections += communitySection
            remainingTokens -= communitySectionTokens
        }

        // local context (entities/relationships/claims)
        val localProp = 1.0 - communityProp - textUnitProp
        val localTokens = (maxContextTokens * localProp).toInt()
        val (entitySection, entityTokens) =
            buildEntitySection(
                selectedEntities = selectedEntities,
                includeEntityRank = includeEntityRank,
                tokenBudget = localTokens,
            )
        if (entitySection.isNotBlank()) {
            sections += entitySection
            remainingTokens -= entityTokens
        }

        val (relationshipSection, relationshipTokens) =
            buildRelationshipSection(
                selectedEntities = selectedEntities,
                includeRelationshipWeight = includeRelationshipWeight,
                relationshipRankingAttribute = relationshipRankingAttribute,
                topK = topKRelationships,
                tokenBudget = localTokens - entityTokens,
            )
        if (relationshipSection.isNotBlank()) {
            sections += relationshipSection
            remainingTokens -= relationshipTokens
        }

        val (claimSection, claimTokens) =
            buildClaimSection(
                selectedEntities = selectedEntities,
                topK = topKClaims,
                tokenBudget = localTokens - entityTokens - relationshipTokens,
            )
        if (claimSection.isNotBlank()) {
            sections += claimSection
            remainingTokens -= claimTokens
        }

        var remainingLocalTokens = localTokens - entityTokens - relationshipTokens - claimTokens
        for ((covType, covariateList) in covariates) {
            if (remainingLocalTokens <= 0) break
            val (covSection, covTokens) = buildCovariateSection(covType, covariateList, selectedEntities, remainingLocalTokens)
            if (covSection.isNotBlank()) {
                sections += covSection
                remainingTokens -= covTokens
                remainingLocalTokens -= covTokens
            }
        }

        // sources
        val textUnitTokens = remainingTokens.coerceAtMost((maxContextTokens * textUnitProp).toInt())
        val sourceResult = buildSourceSection(selectedEntities, textUnitTokens)
        if (sourceResult.section.isNotBlank()) {
            sections += sourceResult.section
            usedChunks += sourceResult.chunks
        }

        val contextText = sections.joinToString(separator = "\n\n")
        val finalChunks = if (usedChunks.isNotEmpty()) usedChunks else selectedEntities.map { entityChunk(it) }
        return LocalContextResult(contextText = contextText, contextChunks = finalChunks)
    }

    private suspend fun mapQueryToEntities(
        query: String,
        includeEntityNames: List<String>,
        excludeEntityNames: List<String>,
        k: Int,
        oversample: Int,
    ): List<Entity> {
        val includeSet = includeEntityNames.map { it.lowercase() }.toSet()
        val excludeSet = excludeEntityNames.map { it.lowercase() }.toSet()
        val entityById = entities.associateBy { it.id }
        val entityByName = entities.associateBy { it.name.lowercase() }

        val seed = embed(query) ?: return emptyList()
        val nearest =
            vectorStore
                .nearestEntities(seed, k * oversample)
                .mapNotNull { (entityId, _) -> entityById[entityId] }

        val includeMatches = includeSet.mapNotNull { entityByName[it] }
        val merged =
            (includeMatches + nearest)
                .filterNot { it.name.lowercase() in excludeSet }
                .distinctBy { it.id }
                .take(k)
        return merged
    }

    private fun buildConversationHistorySection(
        conversationHistory: ConversationHistory,
        includeUserOnly: Boolean,
        maxQaTurns: Int,
    ): String {
        val header = "turn$columnDelimiter${"text"}"
        val turns =
            if (includeUserOnly) {
                conversationHistory.turns.filter { it.role == ConversationTurn.Role.USER }
            } else {
                conversationHistory.turns
            }
        val rows =
            turns
                .takeLast(maxQaTurns)
                .mapIndexed { index, turn ->
                    "${index + 1}$columnDelimiter${turn.content}"
                }.joinToString("\n")
        return if (rows.isBlank()) "" else buildSection("Conversation", header, rows)
    }

    private fun buildEntitySection(
        selectedEntities: List<Entity>,
        includeEntityRank: Boolean,
        tokenBudget: Int,
    ): Pair<String, Int> {
        if (selectedEntities.isEmpty()) return "" to 0
        val summariesById = entitySummaries.associateBy { it.entityId }
        val headerParts = mutableListOf("id", "entity", "description")
        if (includeEntityRank) headerParts += "rank"
        val header = headerParts.joinToString(columnDelimiter)
        val builder = StringBuilder("-----Entities-----\n$header\n")
        var tokens = tokenCount(builder.toString())

        for (entity in selectedEntities) {
            val description = summariesById[entity.id]?.summary ?: ""
            val cells = mutableListOf(entity.id, entity.name, description)
            if (includeEntityRank) cells += "0"
            val row = cells.joinToString(columnDelimiter) + "\n"
            val rowTokens = tokenCount(row)
            if (tokens + rowTokens > tokenBudget) break
            builder.append(row)
            tokens += rowTokens
        }
        val text = builder.toString().trimEnd()
        return if (text.isBlank()) "" to 0 else text to tokens
    }

    @Suppress("CyclomaticComplexMethod", "ReturnCount")
    private fun buildRelationshipSection(
        selectedEntities: List<Entity>,
        includeRelationshipWeight: Boolean,
        relationshipRankingAttribute: String,
        topK: Int,
        tokenBudget: Int,
    ): Pair<String, Int> {
        if (selectedEntities.isEmpty() || relationships.isEmpty()) return "" to 0
        val selectedIds = selectedEntities.map { it.id }.toSet()
        val filtered =
            relationships
                .filter { rel -> rel.sourceId in selectedIds || rel.targetId in selectedIds }
                .sortedWith(
                    compareByDescending<Relationship> {
                        when (relationshipRankingAttribute.lowercase()) {
                            "weight" -> it.description?.length ?: 0
                            else -> it.description?.length ?: 0
                        }
                    },
                ).take(topK * selectedEntities.size)

        if (filtered.isEmpty()) return "" to 0
        val headers = mutableListOf("source", "target", "type", "description")
        if (includeRelationshipWeight) headers += "weight"
        val header = headers.joinToString(columnDelimiter)
        val builder = StringBuilder("-----Relationships-----\n$header\n")
        var tokens = tokenCount(builder.toString())

        for (rel in filtered) {
            val rowParts = mutableListOf(rel.sourceId, rel.targetId, rel.type, rel.description ?: "")
            if (includeRelationshipWeight) {
                rowParts += rel.description?.length?.toString() ?: ""
            }
            val row = rowParts.joinToString(columnDelimiter) + "\n"
            val rowTokens = tokenCount(row)
            if (tokens + rowTokens > tokenBudget) break
            builder.append(row)
            tokens += rowTokens
        }
        val text = builder.toString().trimEnd()
        return if (text.isBlank()) "" to 0 else text to tokens
    }

    @Suppress("ReturnCount")
    private fun buildClaimSection(
        selectedEntities: List<Entity>,
        tokenBudget: Int,
        topK: Int = 10,
    ): Pair<String, Int> {
        if (selectedEntities.isEmpty() || claims.isEmpty()) return "" to 0
        val selectedIds = selectedEntities.map { it.id }.toSet()
        val filtered =
            claims
                .filter { claim -> claim.subject in selectedIds || claim.`object` in selectedIds }
                .take(topK)
        if (filtered.isEmpty()) return "" to 0
        val header =
            listOf("subject", "object", "claim_type", "status", "description", "source_text")
                .joinToString(columnDelimiter)
        val builder = StringBuilder("-----Claims-----\n$header\n")
        var tokens = tokenCount(builder.toString())
        for (claim in filtered) {
            val row =
                listOf(
                    claim.subject,
                    claim.`object`,
                    claim.claimType,
                    claim.status,
                    claim.description,
                    claim.sourceText,
                ).joinToString(columnDelimiter) + "\n"
            val rowTokens = tokenCount(row)
            if (tokens + rowTokens > tokenBudget) break
            builder.append(row)
            tokens += rowTokens
        }
        val text = builder.toString().trimEnd()
        return if (text.isBlank()) "" to 0 else text to tokens
    }

    @Suppress("ReturnCount", "LongParameterList")
    private fun buildCovariateSection(
        covariateType: String,
        covariateList: List<Covariate>,
        selectedEntities: List<Entity>,
        tokenBudget: Int,
    ): Pair<String, Int> {
        if (covariateList.isEmpty() || selectedEntities.isEmpty()) return "" to 0
        val selectedIds = selectedEntities.map { it.id }.toSet()
        val filtered = covariateList.filter { it.subjectId in selectedIds }
        if (filtered.isEmpty()) return "" to 0

        val attributeKeys = filtered.firstOrNull()?.attributes?.keys?.toList() ?: emptyList()
        val headerParts = mutableListOf("id", "entity")
        headerParts.addAll(attributeKeys)
        val header = headerParts.joinToString(columnDelimiter)
        val builder = StringBuilder("-----${covariateType}-----\n$header\n")
        var tokens = tokenCount(builder.toString())

        for (covariate in filtered) {
            val cells = mutableListOf(covariate.id, covariate.subjectId)
            attributeKeys.forEach { key -> cells += (covariate.attributes[key] ?: "") }
            val row = cells.joinToString(columnDelimiter) + "\n"
            val rowTokens = tokenCount(row)
            if (tokens + rowTokens > tokenBudget) break
            builder.append(row)
            tokens += rowTokens
        }
        val text = builder.toString().trimEnd()
        return if (text.isBlank()) "" to 0 else text to tokens
    }

    @Suppress("ReturnCount", "LongParameterList", "UnusedParameter")
    private fun buildCommunitySection(
        selectedEntities: List<Entity>,
        tokenBudget: Int,
        useCommunitySummary: Boolean,
        includeRank: Boolean,
        minCommunityRank: Int,
        topK: Int = 5,
    ): Pair<String, Int> {
        if (selectedEntities.isEmpty() || communities.isEmpty() || communityReports.isEmpty()) return "" to 0
        val selectedIds = selectedEntities.map { it.id }.toSet()
        val communityMatches =
            communities
                .filter { it.entityId in selectedIds }
                .groupingBy { it.communityId }
                .eachCount()

        val selected =
            communityReports
                .filter { report -> communityMatches.containsKey(report.communityId) }
                .sortedByDescending { communityMatches[it.communityId] ?: 0 }
                .filter { _ -> true }
                .take(topK)
        if (selected.isEmpty()) return "" to 0

        val headerParts = mutableListOf("report_id", "summary")
        if (includeRank) headerParts += "rank"
        val header = headerParts.joinToString(columnDelimiter)
        val builder = StringBuilder("-----Reports-----\n$header\n")
        var tokens = tokenCount(builder.toString())

        for (report in selected) {
            val summary = report.summary
            val rowParts = mutableListOf(report.communityId.toString(), summary)
            if (includeRank) rowParts += (minCommunityRank).toString()
            val row = rowParts.joinToString(columnDelimiter) + "\n"
            val rowTokens = tokenCount(row)
            if (tokens + rowTokens > tokenBudget) break
            builder.append(row)
            tokens += rowTokens
        }
        val text = builder.toString().trimEnd()
        return if (text.isBlank()) "" to 0 else text to tokens
    }

    private fun buildSourceSection(
        selectedEntities: List<Entity>,
        tokenBudget: Int,
    ): SourceSection {
        val selectedIds = selectedEntities.map { it.id }.toSet()
        val chunkIds =
            selectedEntities
                .map { it.sourceChunkId }
                .toMutableSet()
        relationships
            .filter { rel -> rel.sourceId in selectedIds || rel.targetId in selectedIds }
            .forEach { rel -> chunkIds += rel.sourceChunkId }

        val primaryUnits = textUnits.filter { it.chunkId in chunkIds }
        val candidates =
            if (primaryUnits.isNotEmpty()) {
                primaryUnits.map { it.id to it.text }
            } else {
                selectTextContext(tokenBudget, selectedEntities)
            }
        if (candidates.isEmpty()) return SourceSection("", emptyList())

        val header = "source_id$columnDelimiter${"text"}"
        val builder = StringBuilder("-----Sources-----\n$header\n")
        val chunks = mutableListOf<QueryContextChunk>()
        var tokens = tokenCount(builder.toString())

        for ((id, text) in candidates) {
            val row = "$id$columnDelimiter$text\n"
            val rowTokens = tokenCount(row)
            if (tokens + rowTokens > tokenBudget) break
            builder.append(row)
            tokens += rowTokens
            chunks += QueryContextChunk(id = id, text = text, score = 1.0)
        }

        val section = builder.toString().trimEnd()
        return if (section.isBlank()) SourceSection("", chunks) else SourceSection(section, chunks)
    }

    private fun selectTextContext(
        tokenBudget: Int,
        selectedEntities: List<Entity>,
    ): List<Pair<String, String>> {
        val queryEmbedding =
            if (selectedEntities.isEmpty()) {
                null
            } else {
                // If entities exist but no chunks, fall back to cosine against text embeddings
                null
            }
        val byChunkId = textUnits.associateBy { it.chunkId }
        val scored =
            textEmbeddings
                .mapNotNull { embedding ->
                    val unit = byChunkId[embedding.chunkId] ?: return@mapNotNull null
                    val score = queryEmbedding?.let { cosineSimilarity(it, embedding.vector) } ?: 0.0
                    Triple(unit.id, unit.text, score)
                }.sortedByDescending { it.third }
        val rows = mutableListOf<Pair<String, String>>()
        var tokens = tokenCount("source_id$columnDelimiter${"text"}\n")
        for ((id, text, _) in scored) {
            val candidateRow = "$id$columnDelimiter$text\n"
            val rowTokens = tokenCount(candidateRow)
            if (tokens + rowTokens > tokenBudget) break
            tokens += rowTokens
            rows += id to text
        }
        return rows
    }

    private suspend fun embed(text: String): List<Double>? =
        withContext(Dispatchers.IO) {
            runCatching {
                val response: Response<dev.langchain4j.data.embedding.Embedding> = embeddingModel.embed(text)
                val vector = response.content().vector()
                vector.asList().map { it.toDouble() }
            }.getOrNull()
        }

    private fun tokenCount(text: String): Int = encoding.countTokens(text)

    private fun cosineSimilarity(
        a: List<Double>,
        b: List<Double>,
    ): Double {
        if (a.isEmpty() || b.isEmpty() || a.size != b.size) return 0.0
        var dot = 0.0
        var magA = 0.0
        var magB = 0.0
        for (i in a.indices) {
            dot += a[i] * b[i]
            magA += a[i] * a[i]
            magB += b[i] * b[i]
        }
        val denom = kotlin.math.sqrt(magA) * kotlin.math.sqrt(magB)
        return if (denom == 0.0) 0.0 else dot / denom
    }

    private fun buildSection(
        title: String,
        header: String,
        rows: String,
    ): String = "-----$title-----\n$header\n$rows"

    private fun entityChunk(entity: Entity): QueryContextChunk = QueryContextChunk(id = entity.id, text = entity.name, score = 1.0)

    data class SourceSection(
        val section: String,
        val chunks: List<QueryContextChunk>,
    )

    private val encoding: Encoding by lazy {
        Encodings.newLazyEncodingRegistry().getEncoding(EncodingType.CL100K_BASE)
    }
}
