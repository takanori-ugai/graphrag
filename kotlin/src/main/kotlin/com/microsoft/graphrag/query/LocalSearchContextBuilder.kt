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
        /**
         * Extracts up to the most recent user messages from the conversation.
         *
         * @param maxTurns The maximum number of user turns to include; if more user turns exist only the most
         * recent `maxTurns` are returned.
         * @return A list of user turn contents in chronological order (oldest to newest among the selected
         * turns). If fewer than `maxTurns` user turns exist, returns all user turns.
         */
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

    /**
     * Builds a local search context for a query by assembling prioritized sections (conversation history,
     * communities, entities, relationships, claims, covariates, and text sources) within a token budget.
     *
     * The builder maps the query to entities (optionally including recent conversation turns), allocates tokens
     * across community, local, and text-source sections according to proportions, and returns the assembled
     * context text, context chunks, and per-section context records and token metrics.
     *
     * @param query The user's query used to find relevant entities and text.
     * @param conversationHistory Optional conversation history to include; used when enriching the query and
     * optionally added as a context section.
     * @param includeEntityNames Exact entity names to force-include in the mapped entity set (case-insensitive).
     * @param excludeEntityNames Exact entity names to exclude from the mapped entity set (case-insensitive).
     * @param conversationHistoryMaxTurns Maximum number of recent turns to consider from conversation history.
     * @param conversationHistoryUserTurnsOnly If true, only user turns (and associated assistant answers when
     * formatting) are considered when extracting history.
     * @param conversationHistoryRecencyBias If true, prefers more recent turns when enforcing the max-turn limit.
     * @param maxContextTokens Total token budget available for the assembled context.
     * @param textUnitProp Fraction of maxContextTokens reserved for source/text-unit content (0.0–1.0).
     * @param communityProp Fraction of maxContextTokens reserved for community content (0.0–1.0). Must satisfy
     * communityProp + textUnitProp <= 1.0.
     * @param topKMappedEntities Max number of entities to retrieve for local context mapping.
     * @param topKRelationships Max number of relationships to include per selected entities (used as a
     * per-entity budget multiplier in some filters).
     * @param topKClaims Max number of claims to include.
     * @param topKCommunities Max number of communities to include.
     * @param includeCommunityRank If true, includes community rank values in the community section output.
     * @param includeEntityRank If true, includes entity rank values in the entity section output.
     * @param includeRelationshipWeight If true, includes relationship weight values in the relationship section output.
     * @param relationshipRankingAttribute Attribute name used to rank relationships (e.g., "rank", "weight", or
     * a numeric attribute key).
     * @param useCommunitySummary If true, uses the community summary text when building community sections.
     * @param minCommunityRank Minimum community rank threshold for including a community report.
     * @param returnCandidateContext If true, populates contextRecords with candidate items for sections even
     * when not all candidates fit into the final context; marks which candidates were included via the
     * `in_context` flag.
     * @param contextCallbacks Optional callbacks invoked with the final LocalContextResult before it is returned.
     *
     * @return A LocalContextResult containing the assembled context text, selected context chunks,
     * contextRecords broken down by section, and token/LLM metrics for the build operation.
     */
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

    /**
     * Finds up to `k` entities relevant to `query` by embedding the query and nearest-neighbor lookup,
     * while forcing inclusion and exclusion by name.
     *
     * The function obtains an embedding for `query` and retrieves `k * oversample` nearest entities
     * from the local vector store, then merges those neighbors with any entities whose names match
     * `includeEntityNames`. Name matching for both include and exclude lists is case-insensitive.
     * Results are de-duplicated by entity id, filtered by the exclude list, and truncated to `k`.
     *
     * @param query The text query to map to entities.
     * @param includeEntityNames Entity names that should always be considered for inclusion (case-insensitive).
     * @param excludeEntityNames Entity names to exclude from the final results (case-insensitive).
     * @param k The maximum number of entities to return.
     * @param oversample The multiplier used to fetch extra nearest neighbors (fetches `k * oversample` then narrows).
     * @return A list of up to `k` matched entities; returns an empty list if the query embedding cannot be obtained.
     */
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

    /**
     * Builds a formatted "Conversation" section from conversation history containing paired user turns and
     * assistant answers.
     *
     * @param conversationHistory The conversation history to extract turns from.
     * @param includeUserOnly When true, include only user turns and omit assistant answers.
     * @param maxQaTurns Maximum number of user–answer pairs to include (<= 0 means no limit).
     * @param recencyBias When true, prefer most recent QA pairs first; otherwise preserve original order.
     * @return The formatted conversation section (including header) or an empty string if there are no rows to include.
     */
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

    /**
     * Builds the "Entities" section text and reports the number of tokens it consumes.
     *
     * Builds a delimited table of selected entities (id, name, description, optional rank and attributes),
     * appending rows until the provided token budget is reached. Populates `contextRecords["entities"]`
     * with included rows; if `returnCandidateContext` is true, populates that key with candidate records
     * for all considered entities with an `"in_context"` flag indicating inclusion.
     *
     * @param selectedEntities Entities to consider for inclusion in the section.
     * @param includeEntityRank If true, include a `rank` column in the header and rows.
     * @param tokenBudget Maximum number of tokens allowed for this section; rows that would exceed the
     *   budget are omitted.
     * @param returnCandidateContext If true, write candidate records for all considered entities with
     *   `"in_context"` set to `"true"` for included rows and `"false"` otherwise.
     * @param contextRecords Mutable map that will receive the section's record list under the `"entities"` key.
     * @return A Pair whose first element is the section text (empty string if no rows were added) and whose
     *   second element is the number of tokens consumed by the returned section.
     */
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

    /**
     * Builds a formatted "Relationships" section constrained by a token budget and returns the section text and
     * tokens used.
     *
     * Populates contextRecords["relationships"] with either the included relationship rows (marked `in_context = true`)
     * or candidate relationship records (marked `in_context = false` or true depending on inclusion) when
     * `returnCandidateContext`
     * is set. Respects `tokenBudget` and may return an empty section if no relationships fit or none are available.
     *
     * @param selectedEntities Entities considered as the focal set for relationship selection.
     * @param includeRelationshipWeight If true, include the relationship `weight` column in the output and records.
     * @param relationshipRankingAttribute Attribute name used to rank relationships when selecting candidates.
     * @param topK Maximum number of relationships to consider per selected entity for the filtered selection.
     * @param tokenBudget Maximum allowed tokens for the built section; the function will stop adding rows when
     * exceeded.
     * @param returnCandidateContext When true, populate `contextRecords["relationships"]` with candidate
     * records and mark which are in-context.
     * @param contextRecords Mutable map that will be updated with relationship records under the
     * "relationships" key when applicable.
     * @param selectedIdentifiers Set of entity identifiers used to determine which relationships are relevant.
     *
     * @return A Pair whose first element is the formatted relationships section (empty string if none) and
     * whose second element is the token count used.
     */
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
                        buildRelationshipRecord(
                            rel = rel,
                            headers = headers,
                            includeWeight = includeRelationshipWeight,
                            attributeKeys = attributeKeys,
                            inContext = false,
                        )
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
                        buildRelationshipRecord(
                            rel = rel,
                            headers = headers,
                            includeWeight = includeRelationshipWeight,
                            attributeKeys = attributeKeys,
                            inContext = false,
                        )
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
                        buildRelationshipRecord(
                            rel = rel,
                            headers = headers,
                            includeWeight = includeRelationshipWeight,
                            attributeKeys = attributeKeys,
                            inContext = false,
                        )
                    }
                contextRecords["relationships"] = candidateRecords.toMutableList()
            }
            return "" to 0
        }

        for (rel in filtered) {
            val rowParts = relationshipRowParts(rel, includeRelationshipWeight, attributeKeys)
            val row = rowParts.joinToString(columnDelimiter) + "\n"
            val rowTokens = tokenCount(row)
            if (tokens + rowTokens > tokenBudget) break
            builder.append(row)
            tokens += rowTokens
            val record =
                buildRelationshipRecord(
                    rel = rel,
                    headers = headers,
                    includeWeight = includeRelationshipWeight,
                    attributeKeys = attributeKeys,
                    inContext = true,
                )
            includedIds += rel.id ?: rel.shortId.orEmpty()
            records += record
        }
        val text = builder.toString().trimEnd()
        if (records.isNotEmpty()) contextRecords["relationships"] = records.toMutableList()
        if (returnCandidateContext) {
            val candidateRecords =
                candidateRelationships.map { rel ->
                    buildRelationshipRecord(
                        rel = rel,
                        headers = headers,
                        includeWeight = includeRelationshipWeight,
                        attributeKeys = attributeKeys,
                        inContext = (rel.id ?: rel.shortId.orEmpty()) in includedIds,
                    )
                }
            contextRecords["relationships"] = candidateRecords.toMutableList()
        }
        return if (text.isBlank()) "" to 0 else text to tokens
    }

    /**
     * Selects and orders relationships relevant to the given set of entity identifiers, preferring links that
     * connect two selected entities and then the highest-priority external links.
     *
     * @param selectedIdentifiers The set of entity ids to consider as the focal network.
     * @param topK Number of external relationships to include per selected entity (total external budget =
     * `topK * selectedIdentifiers.size`).
     * @param rankingAttribute The attribute name used to score and break ties when ordering relationships (for
     * example `"rank"` or `"weight"`).
     * @return A list containing all relationships between two selected entities (sorted by ranking), followed
     * by up to `topK * selectedIdentifiers.size` external relationships (sorted by external endpoint
     * connectivity and then by the ranking attribute).
     */
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

    /**
     * Build the ordered list of column values for a relationship row.
     *
     * The returned list contains, in order: short id (or id), source id, target id, type, description,
     * optionally the weight (when `includeWeight` is true), followed by attribute values in the same
     * order as `attributeKeys`.
     *
     * @param rel The relationship to serialize into row parts.
     * @param includeWeight If true, include the relationship's weight (empty string if missing).
     * @param attributeKeys Keys whose corresponding attribute values should be appended in order.
     * @return A list of string values representing the relationship row columns.
     */
    private fun relationshipRowParts(
        rel: Relationship,
        includeWeight: Boolean,
        attributeKeys: List<String>,
    ): List<String> {
        val rowParts =
            mutableListOf(
                rel.shortId ?: rel.id.orEmpty(),
                rel.sourceId,
                rel.targetId,
                rel.type,
                rel.description ?: "",
            )
        if (includeWeight) rowParts += (rel.weight?.toString() ?: "")
        attributeKeys.forEach { key -> rowParts += (rel.attributes[key] ?: "") }
        return rowParts
    }

    /**
     * Builds a mutable record map representing a relationship row, keyed by the provided headers, and marks it
     * with an `in_context` flag.
     *
     * @param rel The Relationship to convert into row values.
     * @param headers The header names that will be used as keys in the resulting map; order must match the row
     * parts produced.
     * @param includeWeight When true, include the relationship's weight as one of the row parts.
     * @param attributeKeys Additional relationship attribute keys whose values will be appended to the row
     * parts in order.
     * @param inContext If true, sets the "in_context" key to "true"; otherwise sets it to "false".
     * @return A mutable map from header names to string values for the relationship row, with an added
     * "in_context" entry.
     */
    private fun buildRelationshipRecord(
        rel: Relationship,
        headers: List<String>,
        includeWeight: Boolean,
        attributeKeys: List<String>,
        inContext: Boolean,
    ): MutableMap<String, String> =
        headers.zip(relationshipRowParts(rel, includeWeight, attributeKeys)).toMap().toMutableMap().apply {
            this["in_context"] = if (inContext) "true" else "false"
        }

    /**
     * Get the link count for the endpoint of a relationship that lies outside the given set of selected entity IDs.
     *
     * @param relationship The relationship whose external endpoint is queried.
     * @param selectedIds Set of entity IDs considered "selected".
     * @param linkCounts Map from entity ID to its link count.
     * @return The link count for the relationship endpoint not in `selectedIds`, or `0` if not present in `linkCounts`.
     */
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

    /**
     * Compute a numeric ranking value for a relationship based on the specified ranking attribute.
     *
     * @param relationship The relationship whose ranking value is being extracted.
     * @param rankingAttribute The attribute name to use for ranking; supported values are `"rank"`, `"weight"`,
     * or any key present in the relationship's `attributes` map.
     * @return The numeric value of the chosen ranking attribute, or `0.0` if the attribute is missing or cannot
     * be parsed as a number.
     */
    private fun relationshipRankingValue(
        relationship: Relationship,
        rankingAttribute: String,
    ): Double =
        when (rankingAttribute.lowercase()) {
            "rank" -> relationship.rank ?: 0.0
            "weight" -> relationship.weight ?: 0.0
            else -> relationship.attributes[rankingAttribute]?.toDoubleOrNull() ?: 0.0
        }

    /**
     * Produces candidate relationships that touch any of the provided entities, ordered by relevance.
     *
     * Returns relationships that involve any of the given entities, sorted first by the number of distinct connections
     * those relationships make to other (non-selected) entities and then by the numeric value of the specified
     * ranking attribute.
     *
     * @param selectedEntities Entities used to select and prioritize candidate relationships.
     * @param rankingAttribute Name of the relationship attribute used as a secondary numeric sort key (higher
     * values rank earlier).
     * @return A list of relationships touching the selected entities, sorted by (1) link count to non-selected
     * endpoints (descending) and (2) ranking attribute value (descending).
     */
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

    /**
     * Convert an entity's rank (or a fallback relationship-derived rank) into its string form.
     *
     * @param entity The entity whose rank is required.
     * @return The entity's rank as a string; uses `entity.rank` when present, otherwise the count of
     * relationships involving the entity converted to a string.
     */
    private fun entityRank(entity: Entity): String {
        val rankValue = entity.rank ?: relationshipCount(entity.id).toDouble()
        return rankValue.toString()
    }

    /**
     * Count how many relationships reference the given entity ID as either source or target.
     *
     * @param entityId The identifier of the entity to count relationships for.
     * @return The number of relationships that include the entity as a source or target.
     */
    private fun relationshipCount(entityId: String): Int =
        relationships.count { rel -> rel.sourceId == entityId || rel.targetId == entityId }

    /**
     * Builds the "Claims" section containing claims that reference any of the selected entities, constrained by
     * a token budget.
     *
     * @param selectedEntities Entities considered for inclusion (used to short-circuit when empty).
     * @param tokenBudget Maximum number of tokens allowed for the returned section.
     * @param topK Maximum number of matching claims to consider.
     * @param contextRecords Mutable map that will be populated with claim records (key "claims"); records
     * included in the section are marked with `in_context = "true"`. If `returnCandidateContext` is true, this
     * will contain all candidate claims with `in_context` flags set.
     * @param returnCandidateContext When true, populate `contextRecords["claims"]` with all candidate claims
     * (not only those that fit in the token budget) and mark which were included.
     * @param selectedIdentifiers Set of entity identifiers used to filter claims by subject or object.
     * @return Pair of the formatted claims section text and the number of tokens consumed by that section;
     * returns an empty string and 0 if no claims are included.
     */
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

    /**
     * Builds a covariate section listing covariates for the given selected entities under a token budget.
     *
     * The section is formatted with a header row of column names followed by covariate rows until the token
     * budget is exhausted.
     *
     * @param covariateType The display name used for the section header and the key under which records are
     * stored in `contextRecords`.
     * @param covariateList The list of covariates to filter and format.
     * @param selectedEntities Entities used to select relevant covariates (matched by id or name).
     * @param tokenBudget Maximum tokens allowed for the produced section; rows that would exceed this budget
     * are omitted.
     * @param returnCandidateContext If true, `contextRecords` is populated with candidate records for all
     * matching covariates, each annotated with an `in_context` flag.
     * @param contextRecords Mutable map that will be updated with per-covariate records when any rows are
     * included or when `returnCandidateContext` is true.
     * @return A pair containing the formatted covariate section text and the token count consumed by that
     * section; returns an empty string and 0 tokens when no covariates are included.
     */
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

    /**
     * Builds a community reports section listing community reports that match the provided entities within a
     * token budget.
     *
     * The section contains a header and one row per included community report; matching reports are selected by
     * membership and minimum rank, ordered by number of matched entities and report rank, and truncated when
     * the token budget is reached. When requested, candidate records for all matching reports (with
     * `in_context` flags) are written to `contextRecords["reports"]`.
     *
     * @param selectedEntities Entities used to determine which communities match.
     * @param tokenBudget Maximum number of tokens allowed for the returned section text.
     * @param useCommunitySummary If true, include each report's summary in the section; otherwise include the
     * full content when available.
     * @param includeRank If true, append the report rank as a column.
     * @param minCommunityRank Minimum report rank required for inclusion.
     * @param topK Maximum number of community reports to consider for inclusion.
     * @param returnCandidateContext If true, populate `contextRecords["reports"]` with candidate entries for
     * all matching reports and mark which were included (`in_context` = "true"/"false").
     * @param contextRecords Mutable map that will be updated with per-section record entries under the
     * "reports" key when any candidate or included records exist.
     * @return A pair where the first element is the formatted reports section (empty string if nothing was
     * included) and the second element is the token count of that section.
     */
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

    /**
     * Builds the "Sources" section listing source IDs and text snippets relevant to the provided entities,
     * constrained by a token budget, and collects corresponding query context chunks and context records.
     *
     * The method selects primary text units associated with the given entities (or falls back to text units
     * linked by entity-related relationships or an embedding-based fallback), formats rows as `source_id|text`,
     * and includes as many rows as fit within `tokenBudget`. For each included row a QueryContextChunk and a
     * context record with `in_context = "true"` are produced. If `returnCandidateContext` is true, the
     * contextRecords entry for "sources" will contain all candidate sources with `in_context` set to `"true"`
     * for included rows and `"false"` for excluded candidates.
     *
     * @param selectedEntities Entities to derive relevant sources from.
     * @param tokenBudget Maximum allowed token count for the resulting section (header + rows).
     * @param returnCandidateContext When true, populate `contextRecords["sources"]` with all candidate sources
     *   and an `in_context` flag indicating which were included in the final section.
     * @param contextRecords Mutable map that will be updated with source context records under the key "sources".
     * @return A SourceSection containing the formatted section string (empty if no rows fit) and the list of
     *   QueryContextChunk objects for each included source.
     */
    private suspend fun buildSourceSection(
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

    /**
     * Selects text units most relevant to the provided entities, formatted as (sourceId, text) rows that fit
     * within a token budget.
     *
     * @param tokenBudget Maximum number of tokens allowed for the assembled source rows (including header).
     * @param selectedEntities Entities whose names are used to form a query embedding for relevance scoring;
     * when empty, units are treated with equal (zero) relevance.
     * @return Ordered list of (sourceId, text) pairs representing the selected text units, sorted by descending
     * relevance and truncated to respect the token budget.
     */
    private suspend fun selectTextContext(
        tokenBudget: Int,
        selectedEntities: List<Entity>,
    ): List<Pair<String, String>> {
        val queryEmbedding =
            if (selectedEntities.isNotEmpty()) {
                val entityText = selectedEntities.joinToString(" ") { it.name }
                embed(entityText)
            } else {
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

    /**
     * Requests an embedding for the given text from the embedding model.
     *
     * @return A list of doubles representing the embedding vector, or `null` if an embedding could not be obtained.
     */
    private suspend fun embed(text: String): List<Double>? =
        withContext(Dispatchers.IO) {
            runCatching {
                val response: Response<dev.langchain4j.data.embedding.Embedding> = embeddingModel.embed(text)
                val vector = response.content().vector()
                vector.asList().map { it.toDouble() }
            }.getOrNull()
        }

    /**
     * Counts the number of tokens in the given text.
     *
     * @return The number of tokens in the text.
     */
    private fun tokenCount(text: String): Int = encoding.countTokens(text)

    /**
     * Compute the cosine similarity between two numeric vectors.
     *
     * Returns a value between -1.0 and 1.0 that measures the directional similarity of the vectors;
     * 1.0 indicates identical direction, -1.0 indicates opposite direction.
     * If either input is empty, lengths differ, or either vector has zero magnitude, returns 0.0.
     *
     * @param a First vector.
     * @param b Second vector.
     * @return Cosine similarity score between `a` and `b`, or 0.0 for invalid or zero-magnitude inputs.
     */
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

    /**
     * Formats a section block with a delimited title, a header line, and row content.
     *
     * @return The concatenated section string in the form:
     * "-----<title>-----\n<header>\n<rows>"
     */
    private fun buildSection(
        title: String,
        header: String,
        rows: String,
    ): String = "-----$title-----\n$header\n$rows"

    /**
     * Creates a QueryContextChunk representing the given entity for inclusion in query context.
     *
     * @param entity The entity to convert into a context chunk.
     * @return A QueryContextChunk with the entity's id as `id`, the entity's name as `text`, and a score of `1.0`.
     */
    private fun entityChunk(entity: Entity): QueryContextChunk = QueryContextChunk(id = entity.id, text = entity.name, score = 1.0)

    data class SourceSection(
        val section: String,
        val chunks: List<QueryContextChunk>,
    )

    private val encoding: Encoding by lazy {
        Encodings.newLazyEncodingRegistry().getEncoding(EncodingType.CL100K_BASE)
    }
}
