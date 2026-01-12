package com.microsoft.graphrag.query

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingType
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
        val contextRecords: Map<String, List<MutableMap<String, String>>>,
        val llmCalls: Int = 0,
        val promptTokens: Int = 0,
        val outputTokens: Int = 0,
        val llmCallsCategories: Map<String, Int> = emptyMap(),
        val promptTokensCategories: Map<String, Int> = emptyMap(),
        val outputTokensCategories: Map<String, Int> = emptyMap(),
    )

    @Suppress("LongMethod", "ReturnCount", "LongParameterList", "CyclomaticComplexMethod")
    suspend fun buildContext(
        query: String,
        conversationHistory: ConversationHistory? = null,
        includeEntityNames: List<String> = emptyList(),
        excludeEntityNames: List<String> = emptyList(),
        conversationHistoryMaxTurns: Int = 5,
        conversationHistoryUserTurnsOnly: Boolean = true,
        conversationHistoryRecencyBias: Boolean = true,
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
        returnCandidateContext: Boolean = false,
        contextCallbacks: List<(LocalContextResult) -> Unit> = emptyList(),
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
        val contextRecords = mutableMapOf<String, MutableList<MutableMap<String, String>>>()
        var remainingTokens = maxContextTokens
        val selectedIdentifiers =
            selectedEntities
                .flatMap { listOf(it.id, it.name) }
                .toSet()

        // conversation history context
        if (conversationHistory != null && conversationHistory.turns.isNotEmpty()) {
            val historySection =
                buildConversationHistorySection(
                    conversationHistory = conversationHistory,
                    includeUserOnly = conversationHistoryUserTurnsOnly,
                    maxQaTurns = conversationHistoryMaxTurns,
                    recencyBias = conversationHistoryRecencyBias,
                )
            val tokens = tokenCount(historySection)
            if (tokens <= remainingTokens) {
                sections += historySection
                remainingTokens -= tokens
            }
        }

        // community context
        val communityTokens = (remainingTokens * communityProp).toInt().coerceAtLeast(0)
        val (communitySection, communitySectionTokens) =
            buildCommunitySection(
                selectedEntities,
                tokenBudget = communityTokens,
                useCommunitySummary = useCommunitySummary,
                includeRank = includeCommunityRank,
                minCommunityRank = minCommunityRank,
                topK = topKCommunities,
                returnCandidateContext = returnCandidateContext,
                contextRecords = contextRecords,
            )
        if (communitySection.isNotBlank()) {
            sections += communitySection
            remainingTokens -= communitySectionTokens
        }

        // local context (entities/relationships/claims)
        val localProp = 1.0 - communityProp - textUnitProp
        val localTokens =
            minOf(remainingTokens, (maxContextTokens * localProp).toInt()).coerceAtLeast(0)
        val (entitySection, entityTokens) =
            buildEntitySection(
                selectedEntities = selectedEntities,
                includeEntityRank = includeEntityRank,
                tokenBudget = localTokens,
                returnCandidateContext = returnCandidateContext,
                contextRecords = contextRecords,
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
                returnCandidateContext = returnCandidateContext,
                contextRecords = contextRecords,
                selectedIdentifiers = selectedIdentifiers,
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
                contextRecords = contextRecords,
                returnCandidateContext = returnCandidateContext,
                selectedIdentifiers = selectedIdentifiers,
            )
        if (claimSection.isNotBlank()) {
            sections += claimSection
            remainingTokens -= claimTokens
        }

        var remainingLocalTokens = localTokens - entityTokens - relationshipTokens - claimTokens
        for ((covType, covariateList) in covariates) {
            if (remainingLocalTokens <= 0) break
            val (covSection, covTokens) =
                buildCovariateSection(
                    covType,
                    covariateList,
                    selectedEntities,
                    remainingLocalTokens,
                    returnCandidateContext,
                    contextRecords,
                )
            if (covSection.isNotBlank()) {
                sections += covSection
                remainingTokens -= covTokens
                remainingLocalTokens -= covTokens
                if (returnCandidateContext) {
                    // mark in-context covariates
                    contextRecords.getOrPut(covType.lowercase()) { mutableListOf() }.forEach { it["in_context"] = "true" }
                }
            }
        }

        // sources
        val textUnitTokens =
            remainingTokens
                .coerceAtLeast(0)
                .coerceAtMost((maxContextTokens * textUnitProp).toInt())
        val sourceResult = buildSourceSection(selectedEntities, textUnitTokens, returnCandidateContext, contextRecords)
        if (sourceResult.section.isNotBlank()) {
            sections += sourceResult.section
            usedChunks += sourceResult.chunks
        }

        val contextText = sections.joinToString(separator = "\n\n")
        val finalChunks = if (usedChunks.isNotEmpty()) usedChunks else selectedEntities.map { entityChunk(it) }
        val promptTokens = tokenCount(contextText)
        val result =
            LocalContextResult(
                contextText = contextText,
                contextChunks = finalChunks,
                contextRecords = contextRecords,
                llmCalls = 0,
                promptTokens = promptTokens,
                outputTokens = 0,
                llmCallsCategories = mapOf("build_context" to 0),
                promptTokensCategories = mapOf("build_context" to promptTokens),
                outputTokensCategories = mapOf("build_context" to 0),
            )
        contextCallbacks.forEach { it(result) }
        return result
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
        recencyBias: Boolean,
    ): String {
        val header = "turn$columnDelimiter${"content"}"
        var qaTurns =
            buildList {
                var currentUser: ConversationTurn? = null
                val buffer = mutableListOf<ConversationTurn>()
                for (turn in conversationHistory.turns) {
                    if (turn.role == ConversationTurn.Role.USER) {
                        if (currentUser != null) add(QATurn(user = currentUser, answers = buffer.toList()))
                        currentUser = turn
                        buffer.clear()
                    } else {
                        buffer += turn
                    }
                }
                if (currentUser != null) add(QATurn(user = currentUser, answers = buffer.toList()))
            }
        if (includeUserOnly) {
            qaTurns = qaTurns.map { QATurn(user = it.user, answers = emptyList()) }
        }
        if (recencyBias) qaTurns = qaTurns.asReversed()
        if (maxQaTurns > 0 && qaTurns.size > maxQaTurns) qaTurns = qaTurns.take(maxQaTurns)

        val rows =
            buildString {
                qaTurns.forEach { turn ->
                    appendLine("${ConversationTurn.Role.USER.name.lowercase()}$columnDelimiter${turn.user.content}")
                    turn.answers.forEach { ans ->
                        appendLine("${ConversationTurn.Role.ASSISTANT.name.lowercase()}$columnDelimiter${ans.content}")
                    }
                }
            }.trimEnd()
        return if (rows.isBlank()) "" else buildSection("Conversation", header, rows)
    }

    private data class QATurn(
        val user: ConversationTurn,
        val answers: List<ConversationTurn>,
    )

    private fun buildEntitySection(
        selectedEntities: List<Entity>,
        includeEntityRank: Boolean,
        tokenBudget: Int,
        returnCandidateContext: Boolean,
        contextRecords: MutableMap<String, MutableList<MutableMap<String, String>>>,
    ): Pair<String, Int> {
        if (selectedEntities.isEmpty()) return "" to 0
        val sortedEntities =
            selectedEntities.sortedByDescending { entity ->
                entityRank(entity).toDoubleOrNull() ?: 0.0
            }
        val summariesById = entitySummaries.associateBy { it.entityId }
        val attributeKeys =
            sortedEntities
                .firstOrNull()
                ?.attributes
                ?.keys
                ?.toList() ?: emptyList()
        val headerParts = mutableListOf("id", "entity", "description")
        if (includeEntityRank) headerParts += "rank"
        headerParts.addAll(attributeKeys)
        val header = headerParts.joinToString(columnDelimiter)
        val builder = StringBuilder("-----Entities-----\n$header\n")
        var tokens = tokenCount(builder.toString())
        val includedIds = mutableSetOf<String>()
        val records = mutableListOf<MutableMap<String, String>>()

        for (entity in sortedEntities) {
            val description = summariesById[entity.id]?.summary ?: entity.description.orEmpty()
            val cells = mutableListOf(entity.shortId ?: entity.id, entity.name, description)
            if (includeEntityRank) cells += entityRank(entity)
            attributeKeys.forEach { key -> cells += (entity.attributes[key] ?: "") }
            val row = cells.joinToString(columnDelimiter) + "\n"
            val rowTokens = tokenCount(row)
            if (tokens + rowTokens > tokenBudget) break
            builder.append(row)
            tokens += rowTokens
            includedIds += entity.id
            records +=
                headerParts
                    .zip(cells)
                    .toMap()
                    .toMutableMap()
                    .apply { this["in_context"] = "true" }
        }
        val text = builder.toString().trimEnd()
        if (records.isNotEmpty()) contextRecords["entities"] = records.toMutableList()

        if (returnCandidateContext) {
            val candidateRecords =
                sortedEntities.map { entity ->
                    val description = summariesById[entity.id]?.summary ?: entity.description.orEmpty()
                    val cells = mutableListOf(entity.shortId ?: entity.id, entity.name, description)
                    if (includeEntityRank) cells += entityRank(entity)
                    attributeKeys.forEach { key -> cells += (entity.attributes[key] ?: "") }
                    val record = headerParts.zip(cells).toMap().toMutableMap()
                    record["in_context"] = if (entity.id in includedIds) "true" else "false"
                    record
                }
            contextRecords["entities"] = candidateRecords.toMutableList()
        }
        return if (text.isBlank()) "" to 0 else text to tokens
    }

    @Suppress("CyclomaticComplexMethod", "ReturnCount")
    private fun buildRelationshipSection(
        selectedEntities: List<Entity>,
        includeRelationshipWeight: Boolean,
        relationshipRankingAttribute: String,
        topK: Int,
        tokenBudget: Int,
        returnCandidateContext: Boolean,
        contextRecords: MutableMap<String, MutableList<MutableMap<String, String>>>,
        selectedIdentifiers: Set<String>,
    ): Pair<String, Int> {
        if (selectedEntities.isEmpty() || relationships.isEmpty()) {
            if (returnCandidateContext) {
                val candidates = sortedRelationshipsForCandidates(selectedEntities, relationshipRankingAttribute)
                val headers = mutableListOf("id", "source", "target", "type", "description")
                if (includeRelationshipWeight) headers += "weight"
                val attributeKeys =
                    candidates
                        .firstOrNull()
                        ?.attributes
                        ?.keys
                        ?.filterNot { key -> headers.contains(key) || key.equals("weight", ignoreCase = true) }
                        ?: emptyList()
                headers.addAll(attributeKeys)
                val candidateRecords =
                    candidates.map { rel ->
                        val rowParts =
                            mutableListOf(
                                rel.shortId ?: rel.id.orEmpty(),
                                rel.sourceId,
                                rel.targetId,
                                rel.type,
                                rel.description ?: "",
                            )
                        if (includeRelationshipWeight) rowParts += (rel.weight?.toString() ?: "")
                        attributeKeys.forEach { key -> rowParts += (rel.attributes[key] ?: "") }
                        val record = headers.zip(rowParts).toMap().toMutableMap()
                        record["in_context"] = "false"
                        record
                    }
                contextRecords["relationships"] = candidateRecords.toMutableList()
            }
            return "" to 0
        }
        val filtered = filterRelationships(selectedIdentifiers, topK, relationshipRankingAttribute)
        if (filtered.isEmpty()) {
            if (returnCandidateContext) {
                val candidates = sortedRelationshipsForCandidates(selectedEntities, relationshipRankingAttribute)
                val headers = mutableListOf("id", "source", "target", "type", "description")
                if (includeRelationshipWeight) headers += "weight"
                val attributeKeys =
                    candidates
                        .firstOrNull()
                        ?.attributes
                        ?.keys
                        ?.filterNot { key -> headers.contains(key) || key.equals("weight", ignoreCase = true) }
                        ?: emptyList()
                headers.addAll(attributeKeys)
                val candidateRecords =
                    candidates.map { rel ->
                        val rowParts =
                            mutableListOf(
                                rel.shortId ?: rel.id.orEmpty(),
                                rel.sourceId,
                                rel.targetId,
                                rel.type,
                                rel.description ?: "",
                            )
                        if (includeRelationshipWeight) rowParts += (rel.weight?.toString() ?: "")
                        attributeKeys.forEach { key -> rowParts += (rel.attributes[key] ?: "") }
                        val record = headers.zip(rowParts).toMap().toMutableMap()
                        record["in_context"] = "false"
                        record
                    }
                contextRecords["relationships"] = candidateRecords.toMutableList()
            }
            return "" to 0
        }

        val headers = mutableListOf("id", "source", "target", "type", "description")
        if (includeRelationshipWeight) headers += "weight"
        val attributeKeys =
            filtered
                .firstOrNull()
                ?.attributes
                ?.keys
                ?.filterNot { key -> headers.contains(key) || key.equals("weight", ignoreCase = true) }
                ?: emptyList()
        headers.addAll(attributeKeys)
        val header = headers.joinToString(columnDelimiter)
        val builder = StringBuilder("-----Relationships-----\n$header\n")
        var tokens = tokenCount(builder.toString())
        val includedIds = mutableSetOf<String>()
        val records = mutableListOf<MutableMap<String, String>>()
        val candidateRelationships =
            if (returnCandidateContext) {
                sortedRelationshipsForCandidates(
                    selectedEntities = selectedEntities,
                    rankingAttribute = relationshipRankingAttribute,
                )
            } else {
                emptyList()
            }
        if (tokenBudget <= tokens) {
            if (returnCandidateContext) {
                val candidateRecords =
                    candidateRelationships.map { rel ->
                        val rowParts =
                            mutableListOf(
                                rel.shortId ?: rel.id.orEmpty(),
                                rel.sourceId,
                                rel.targetId,
                                rel.type,
                                rel.description ?: "",
                            )
                        if (includeRelationshipWeight) rowParts += (rel.weight?.toString() ?: "")
                        attributeKeys.forEach { key -> rowParts += (rel.attributes[key] ?: "") }
                        val record = headers.zip(rowParts).toMap().toMutableMap()
                        record["in_context"] = "false"
                        record
                    }
                contextRecords["relationships"] = candidateRecords.toMutableList()
            }
            return "" to 0
        }

        for (rel in filtered) {
            val rowParts =
                mutableListOf(
                    rel.shortId ?: rel.id.orEmpty(),
                    rel.sourceId,
                    rel.targetId,
                    rel.type,
                    rel.description ?: "",
                )
            if (includeRelationshipWeight) rowParts += (rel.weight?.toString() ?: "")
            attributeKeys.forEach { key -> rowParts += (rel.attributes[key] ?: "") }
            val row = rowParts.joinToString(columnDelimiter) + "\n"
            val rowTokens = tokenCount(row)
            if (tokens + rowTokens > tokenBudget) break
            builder.append(row)
            tokens += rowTokens
            val record = headers.zip(rowParts).toMap().toMutableMap()
            record["in_context"] = "true"
            includedIds += rel.id ?: rel.shortId.orEmpty()
            records += record
        }
        val text = builder.toString().trimEnd()
        if (records.isNotEmpty()) contextRecords["relationships"] = records.toMutableList()
        if (returnCandidateContext) {
            val candidateRecords =
                candidateRelationships.map { rel ->
                    val rowParts =
                        mutableListOf(
                            rel.shortId ?: rel.id.orEmpty(),
                            rel.sourceId,
                            rel.targetId,
                            rel.type,
                            rel.description ?: "",
                        )
                    if (includeRelationshipWeight) rowParts += (rel.weight?.toString() ?: "")
                    attributeKeys.forEach { key -> rowParts += (rel.attributes[key] ?: "") }
                    val record = headers.zip(rowParts).toMap().toMutableMap()
                    record["in_context"] = if ((rel.id ?: rel.shortId.orEmpty()) in includedIds) "true" else "false"
                    record
                }
            contextRecords["relationships"] = candidateRecords.toMutableList()
        }
        return if (text.isBlank()) "" to 0 else text to tokens
    }

    private fun filterRelationships(
        selectedIdentifiers: Set<String>,
        topK: Int,
        rankingAttribute: String,
    ): List<Relationship> {
        if (selectedIdentifiers.isEmpty()) return emptyList()

        val inNetwork =
            relationships
                .filter { rel -> rel.sourceId in selectedIdentifiers && rel.targetId in selectedIdentifiers }
                .sortedByDescending { relationshipRankingValue(it, rankingAttribute) }

        val outNetwork =
            relationships
                .filter { rel ->
                    (rel.sourceId in selectedIdentifiers) xor (rel.targetId in selectedIdentifiers)
                }
        if (outNetwork.isEmpty()) return inNetwork

        val linkCounts =
            outNetwork
                .flatMap { listOf(it.sourceId, it.targetId) }
                .filterNot { it in selectedIdentifiers }
                .groupingBy { it }
                .eachCount()

        val sortedOutNetwork =
            outNetwork.sortedWith(
                compareByDescending<Relationship> { relationshipLinkCount(it, selectedIdentifiers, linkCounts) }
                    .thenByDescending { relationshipRankingValue(it, rankingAttribute) },
            )

        val relationshipBudget = topK * selectedIdentifiers.size
        return inNetwork + sortedOutNetwork.take(relationshipBudget)
    }

    private fun relationshipLinkCount(
        relationship: Relationship,
        selectedIds: Set<String>,
        linkCounts: Map<String, Int>,
    ): Int {
        val outNetworkEntity =
            if (relationship.sourceId !in selectedIds) {
                relationship.sourceId
            } else {
                relationship.targetId
            }
        return linkCounts[outNetworkEntity] ?: 0
    }

    private fun relationshipRankingValue(
        relationship: Relationship,
        rankingAttribute: String,
    ): Double =
        when (rankingAttribute.lowercase()) {
            "rank" -> relationship.rank ?: 0.0
            "weight" -> relationship.weight ?: 0.0
            else -> relationship.attributes[rankingAttribute]?.toDoubleOrNull() ?: 0.0
        }

    private fun sortedRelationshipsForCandidates(
        selectedEntities: List<Entity>,
        rankingAttribute: String,
    ): List<Relationship> {
        val selectedIds =
            selectedEntities
                .flatMap { listOf(it.id, it.name) }
                .toSet()
        val candidate =
            relationships.filter { rel -> rel.sourceId in selectedIds || rel.targetId in selectedIds }
        if (candidate.isEmpty()) return candidate

        val linkCounts =
            candidate
                .flatMap { listOf(it.sourceId, it.targetId) }
                .filterNot { it in selectedIds }
                .groupingBy { it }
                .eachCount()

        return candidate.sortedWith(
            compareByDescending<Relationship> { relationshipLinkCount(it, selectedIds, linkCounts) }
                .thenByDescending { relationshipRankingValue(it, rankingAttribute) },
        )
    }

    private fun entityRank(entity: Entity): String {
        val rankValue = entity.rank ?: relationshipCount(entity.id).toDouble()
        return rankValue.toString()
    }

    private fun relationshipCount(entityId: String): Int =
        relationships.count { rel -> rel.sourceId == entityId || rel.targetId == entityId }

    @Suppress("ReturnCount")
    private fun buildClaimSection(
        selectedEntities: List<Entity>,
        tokenBudget: Int,
        topK: Int = 10,
        contextRecords: MutableMap<String, MutableList<MutableMap<String, String>>>,
        returnCandidateContext: Boolean = false,
        selectedIdentifiers: Set<String>,
    ): Pair<String, Int> {
        if (selectedEntities.isEmpty() || claims.isEmpty()) return "" to 0
        val selectedIds = selectedIdentifiers
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
        val records = mutableListOf<MutableMap<String, String>>()
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
            val record =
                mutableMapOf(
                    "subject" to claim.subject,
                    "object" to claim.`object`,
                    "claim_type" to claim.claimType,
                    "status" to claim.status,
                    "description" to claim.description,
                    "source_text" to claim.sourceText,
                    "in_context" to "true",
                )
            records += record
        }
        val text = builder.toString().trimEnd()
        if (records.isNotEmpty()) contextRecords["claims"] = records.toMutableList()
        // candidate claims retain ordering and mark out-of-context rows
        if (returnCandidateContext) {
            val candidateRecords =
                claims
                    .filter { claim -> claim.subject in selectedIds || claim.`object` in selectedIds }
                    .map { claim ->
                        mutableMapOf(
                            "subject" to claim.subject,
                            "object" to claim.`object`,
                            "claim_type" to claim.claimType,
                            "status" to claim.status,
                            "description" to claim.description,
                            "source_text" to claim.sourceText,
                            "in_context" to
                                if (records.any { it["subject"] == claim.subject && it["object"] == claim.`object` }) "true" else "false",
                        )
                    }
            contextRecords["claims"] = candidateRecords.toMutableList()
        }
        return if (text.isBlank()) "" to 0 else text to tokens
    }

    @Suppress("ReturnCount", "LongParameterList")
    private fun buildCovariateSection(
        covariateType: String,
        covariateList: List<Covariate>,
        selectedEntities: List<Entity>,
        tokenBudget: Int,
        returnCandidateContext: Boolean,
        contextRecords: MutableMap<String, MutableList<MutableMap<String, String>>>,
    ): Pair<String, Int> {
        if (covariateList.isEmpty() || selectedEntities.isEmpty()) return "" to 0
        val selectedIds =
            selectedEntities
                .flatMap { listOf(it.id, it.name) }
                .toSet()
        val filtered = covariateList.filter { it.subjectId in selectedIds }
        if (filtered.isEmpty()) return "" to 0

        val attributeKeys =
            filtered
                .firstOrNull()
                ?.attributes
                ?.keys
                ?.toList() ?: emptyList()
        val headerParts = mutableListOf("id", "entity")
        headerParts.addAll(attributeKeys)
        val header = headerParts.joinToString(columnDelimiter)
        val builder = StringBuilder("-----$covariateType-----\n$header\n")
        var tokens = tokenCount(builder.toString())
        val records = mutableListOf<MutableMap<String, String>>()

        for (covariate in filtered) {
            val cells = mutableListOf(covariate.id, covariate.subjectId)
            attributeKeys.forEach { key -> cells += (covariate.attributes[key] ?: "") }
            val row = cells.joinToString(columnDelimiter) + "\n"
            val rowTokens = tokenCount(row)
            if (tokens + rowTokens > tokenBudget) break
            builder.append(row)
            tokens += rowTokens
            records +=
                headerParts
                    .zip(cells)
                    .toMap()
                    .toMutableMap()
                    .apply { this["in_context"] = "true" }
        }
        val text = builder.toString().trimEnd()
        if (records.isNotEmpty()) contextRecords[covariateType.lowercase()] = records.toMutableList()

        if (returnCandidateContext) {
            val candidateRecords =
                filtered.map { covariate ->
                    val cells = mutableListOf(covariate.id, covariate.subjectId)
                    attributeKeys.forEach { key -> cells += (covariate.attributes[key] ?: "") }
                    val record = headerParts.zip(cells).toMap().toMutableMap()
                    val inContext = records.any { it["id"] == covariate.id && it["in_context"] == "true" }
                    record["in_context"] = if (inContext) "true" else "false"
                    record
                }
            contextRecords[covariateType.lowercase()] = candidateRecords.toMutableList()
        }
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
        returnCandidateContext: Boolean,
        contextRecords: MutableMap<String, MutableList<MutableMap<String, String>>>,
    ): Pair<String, Int> {
        if (selectedEntities.isEmpty() || communities.isEmpty() || communityReports.isEmpty()) return "" to 0
        val selectedIds =
            selectedEntities
                .flatMap { listOf(it.id, it.name) }
                .toSet()
        val communityMatches =
            communities
                .filter { it.entityId in selectedIds }
                .groupingBy { it.communityId }
                .eachCount()

        val selected =
            communityReports
                .filter { report -> communityMatches.containsKey(report.communityId) }
                .filter { report -> (report.rank ?: 0.0) >= minCommunityRank }
                .sortedWith(
                    compareByDescending<CommunityReport> { communityMatches[it.communityId] ?: 0 }
                        .thenByDescending { it.rank ?: 0.0 },
                ).take(topK)
        if (selected.isEmpty()) return "" to 0

        val attributeKeys =
            selected
                .firstOrNull()
                ?.attributes
                ?.keys
                ?.toList() ?: emptyList()
        val headerParts = mutableListOf("id", "title")
        headerParts.addAll(attributeKeys)
        headerParts += if (useCommunitySummary) "summary" else "content"
        if (includeRank) headerParts += "rank"
        val header = headerParts.joinToString(columnDelimiter)
        val builder = StringBuilder("-----Reports-----\n$header\n")
        var tokens = tokenCount(builder.toString())
        val includedIds = mutableSetOf<String>()
        val records = mutableListOf<MutableMap<String, String>>()

        for (report in selected) {
            val content = if (useCommunitySummary) report.summary else report.fullContent ?: report.summary
            val rowParts =
                mutableListOf(
                    report.shortId ?: report.id ?: report.communityId.toString(),
                    report.title ?: report.communityId.toString(),
                )
            attributeKeys.forEach { key -> rowParts += (report.attributes[key] ?: "") }
            rowParts += content
            if (includeRank) rowParts += (report.rank?.toString() ?: "")
            val row = rowParts.joinToString(columnDelimiter) + "\n"
            val rowTokens = tokenCount(row)
            if (tokens + rowTokens > tokenBudget) break
            builder.append(row)
            tokens += rowTokens
            val record = headerParts.zip(rowParts).toMap().toMutableMap()
            record["in_context"] = "true"
            includedIds += report.id ?: report.shortId ?: report.communityId.toString()
            records += record
        }
        val text = builder.toString().trimEnd()
        if (records.isNotEmpty()) contextRecords["reports"] = records.toMutableList()

        if (returnCandidateContext) {
            val candidateRecords =
                communityReports
                    .filter { report -> communityMatches.containsKey(report.communityId) }
                    .sortedWith(
                        compareByDescending<CommunityReport> { communityMatches[it.communityId] ?: 0 }
                            .thenByDescending { it.rank ?: 0.0 },
                    ).map { report ->
                        val content = if (useCommunitySummary) report.summary else report.fullContent ?: report.summary
                        val rowParts =
                            mutableListOf(
                                report.shortId ?: report.id ?: report.communityId.toString(),
                                report.title ?: report.communityId.toString(),
                            )
                        attributeKeys.forEach { key -> rowParts += (report.attributes[key] ?: "") }
                        rowParts += content
                        if (includeRank) rowParts += (report.rank?.toString() ?: "")
                        val record = headerParts.zip(rowParts).toMap().toMutableMap()
                        record["in_context"] =
                            if ((report.id ?: report.shortId ?: report.communityId.toString()) in includedIds) "true" else "false"
                        record
                    }
            contextRecords["reports"] = candidateRecords.toMutableList()
        }
        return if (text.isBlank()) "" to 0 else text to tokens
    }

    private fun buildSourceSection(
        selectedEntities: List<Entity>,
        tokenBudget: Int,
        returnCandidateContext: Boolean,
        contextRecords: MutableMap<String, MutableList<MutableMap<String, String>>>,
    ): SourceSection {
        val selectedIds = selectedEntities.map { it.id }.toSet()
        val chunkIds =
            selectedEntities
                .map { it.sourceChunkId }
                .toMutableSet()
        relationships
            .filter { rel -> rel.sourceId in selectedIds || rel.targetId in selectedIds }
            .forEach { rel -> chunkIds += rel.sourceChunkId }

        val textUnitsById = textUnits.associateBy { it.id }
        val relationshipById =
            relationships.filter { rel -> rel.sourceId in selectedIds || rel.targetId in selectedIds }
        val textUnitRelCounts =
            relationshipById
                .flatMap { rel -> rel.textUnitIds.map { it to rel } }
                .groupingBy { it.first }
                .eachCount()
        val unitInfo =
            selectedEntities
                .flatMapIndexed { index, entity ->
                    entity.textUnitIds
                        .mapNotNull { tuId ->
                            val tu = textUnitsById[tuId]
                            if (tu != null) Triple(tu, index, textUnitRelCounts[tuId] ?: 0) else null
                        }
                }.sortedWith(compareBy<Triple<TextUnit, Int, Int>> { it.second }.thenByDescending { it.third })

        val primaryUnits =
            if (unitInfo.isNotEmpty()) {
                unitInfo.map { it.first }
            } else {
                textUnits.filter { it.chunkId in chunkIds }
            }
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
        val records = mutableListOf<MutableMap<String, String>>()

        for ((id, text) in candidates) {
            val row = "$id$columnDelimiter$text\n"
            val rowTokens = tokenCount(row)
            if (tokens + rowTokens > tokenBudget) break
            builder.append(row)
            tokens += rowTokens
            chunks += QueryContextChunk(id = id, text = text, score = 1.0)
            records += mutableMapOf("source_id" to id, "text" to text, "in_context" to "true")
        }

        val section = builder.toString().trimEnd()
        if (records.isNotEmpty()) contextRecords["sources"] = records.toMutableList()

        if (returnCandidateContext) {
            val candidateRecords =
                candidates.map { (id, text) ->
                    mutableMapOf(
                        "source_id" to id,
                        "text" to text,
                        "in_context" to if (records.any { it["source_id"] == id && it["in_context"] == "true" }) "true" else "false",
                    )
                }
            contextRecords["sources"] = candidateRecords.toMutableList()
        }

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
