package com.microsoft.graphrag.index

import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.service.AiServices
import io.github.oshai.kotlinlogging.KotlinLogging

class CommunityReportWorkflow(
    private val chatModel: OpenAiChatModel,
    private val prompts: PromptRepository = PromptRepository(),
) {
    private val logger = KotlinLogging.logger {}
    private val reporter = AiServices.create(Reporter::class.java, chatModel)

    fun generateReports(
        assignments: List<CommunityAssignment>,
        entities: List<Entity>,
        relationships: List<Relationship>,
        textUnits: List<TextUnit>,
        claims: List<Claim>,
        hierarchy: Map<Int, Int?> = emptyMap(),
        priorReports: List<CommunityReport> = emptyList(),
    ): List<CommunityReport> {
        val byCommunity = assignments.groupBy { it.communityId }
        return byCommunity.map { (communityId, members) ->
            val communityEntities = entities.filter { e -> members.any { it.entityId == e.id } }
            val communityRelationships =
                relationships.filter { rel ->
                    members.any { it.entityId == rel.sourceId || it.entityId == rel.targetId }
                }
            val communityTextUnits =
                textUnitsForCommunity(
                    communityEntities,
                    textUnits,
                )
            val communityClaims = claimsForCommunity(communityEntities, claims)
            val subReports = subReportsForCommunity(communityId, priorReports)
            val summary =
                if (communityEntities.isEmpty()) {
                    "No entities in this community."
                } else {
                    summarizeCommunity(
                        communityId,
                        communityEntities,
                        communityRelationships,
                        communityTextUnits,
                        communityClaims,
                        subReports,
                    )
                }
            CommunityReport(
                communityId = communityId,
                summary = summary,
                parentCommunityId = hierarchy[communityId],
            )
        }
    }

    private fun summarizeCommunity(
        communityId: Int,
        entities: List<Entity>,
        relationships: List<Relationship>,
        textUnits: List<TextUnit>,
        claims: List<Claim>,
        subReports: List<CommunityReport>,
    ): String {
        val mapSummaries = mapStage(communityId, entities, relationships, textUnits, claims)
        if (mapSummaries.isEmpty()) return buildFallbackSummary(communityId, entities, relationships, claims)
        if (mapSummaries.size == 1) {
            val summary = mapSummaries.first().trim()
            return summary.ifBlank { buildFallbackSummary(communityId, entities, relationships, claims) }
        }
        val reduced = reduceStage(communityId, mapSummaries, entities, relationships, claims, subReports).trim()
        return reduced.ifBlank { buildFallbackSummary(communityId, entities, relationships, claims) }
    }

    /**
     * Builds a CSV table of entities with columns "id", "entity", and "description".
     *
     * @param entities List of entities to include; each row uses a 1-based index as the `id`, the entity's `name` as `entity`, and the entity's `description` (or `"entity"` when blank) as `description`. All values are escaped for CSV.
     * @return A string containing the CSV header and one row per entity.
     */
    private fun buildEntitiesTable(entities: List<Entity>): String =
        buildString {
            appendLine("id,entity,description")
            entities.forEachIndexed { idx, e ->
                val description = e.description.orEmpty().ifBlank { "entity" }
                appendLine("${idx + 1},${escapeCsv(e.name)},${escapeCsv(description)}")
            }
        }

    /**
     * Build a CSV table of relationships with header "id,source,target,description".
     *
     * Each relationship becomes a row where `id` is a 1-based row index, `source` and `target` are the relationship endpoint IDs, and `description` is the relationship description or, if absent, the relationship type.
     *
     * @param relationships The relationships to include as CSV rows.
     * @return The CSV string containing the header and one row per relationship.
     */
    private fun buildRelationshipsTable(relationships: List<Relationship>): String =
        buildString {
            appendLine("id,source,target,description")
            relationships.forEachIndexed { idx, r ->
                appendLine(
                    "${idx + 1},${escapeCsv(r.sourceId)},${escapeCsv(r.targetId)}," +
                        escapeCsv(r.description ?: r.type ?: ""),
                )
            }
        }

    /**
     * Builds a CSV-formatted section listing text units with the headers `text_unit_id,chunk_id,text`.
     *
     * Each text unit produces one row; the `text` field is truncated to 500 characters and all fields are CSV-escaped.
     *
     * @param textUnits The list of text units to include as rows.
     * @return A CSV string containing the header line followed by one row per text unit.
     */
    private fun buildTextUnitsSection(textUnits: List<TextUnit>): String =
        buildString {
            appendLine("text_unit_id,chunk_id,text")
            textUnits.forEach { tu ->
                val trimmed = escapeCsv(tu.text.take(500))
                appendLine("${escapeCsv(tu.id)},${escapeCsv(tu.chunkId)},$trimmed")
            }
        }

    private fun mapStage(
        communityId: Int,
        entities: List<Entity>,
        relationships: List<Relationship>,
        textUnits: List<TextUnit>,
        claims: List<Claim>,
    ): List<String> {
        if (textUnits.isEmpty()) {
            val contextText =
                buildString {
                    appendLine("Community $communityId")
                    appendLine("Entities")
                    appendLine(buildEntitiesTable(entities))
                    appendLine()
                    appendLine("Relationships")
                    appendLine(buildRelationshipsTable(relationships))
                    if (claims.isNotEmpty()) {
                        appendLine()
                        appendLine("Claims")
                        appendLine(buildClaimsTable(claims))
                    }
                }
            return listOf(runPrompt(contextText, maxLength = 300))
        }
        val batches = textUnits.chunked(5)
        return batches.map { batch ->
            val contextText =
                buildString {
                    appendLine("Community $communityId")
                    appendLine("Entities")
                    appendLine(buildEntitiesTable(entities))
                    appendLine()
                    appendLine("Relationships")
                    appendLine(buildRelationshipsTable(relationships))
                    appendLine()
                    appendLine("Text Units")
                    appendLine(buildTextUnitsSection(batch))
                    if (claims.isNotEmpty()) {
                        appendLine()
                        appendLine("Claims")
                        appendLine(buildClaimsTable(claims))
                    }
                }
            runPrompt(contextText, maxLength = 220)
        }
    }

    private fun reduceStage(
        communityId: Int,
        mapSummaries: List<String>,
        entities: List<Entity>,
        relationships: List<Relationship>,
        claims: List<Claim>,
        subReports: List<CommunityReport>,
    ): String {
        val contextText =
            buildString {
                appendLine("Community $communityId")
                appendLine("Entities")
                appendLine(buildEntitiesTable(entities))
                appendLine()
                appendLine("Relationships")
                appendLine(buildRelationshipsTable(relationships))
                if (claims.isNotEmpty()) {
                    appendLine()
                    appendLine("Claims")
                    appendLine(buildClaimsTable(claims))
                }
                if (subReports.isNotEmpty()) {
                    appendLine()
                    appendLine("Sub-Community Reports")
                    subReports.forEachIndexed { idx, report ->
                        appendLine("${idx + 1}. [Community ${report.communityId}] ${report.summary.trim()}")
                    }
                }
                appendLine()
                appendLine("Partial Reports")
                mapSummaries.forEachIndexed { idx, summary ->
                    appendLine("${idx + 1}. ${summary.trim()}")
                }
            }
        return runPrompt(contextText, maxLength = 300)
    }

    private fun buildFallbackSummary(
        communityId: Int,
        entities: List<Entity>,
        relationships: List<Relationship>,
        claims: List<Claim>,
    ): String {
        val title = "Community $communityId: ${entities.joinToString { it.name }}"
        val summary =
            buildString {
                append("Entities: ")
                append(entities.joinToString { "${it.name} (${it.type})" })
                if (relationships.isNotEmpty()) {
                    append(". Relationships: ")
                    append(relationships.joinToString { "${it.sourceId} -> ${it.targetId} (${it.type})" })
                }
                if (claims.isNotEmpty()) {
                    append(". Claims: ")
                    append(claims.take(3).joinToString { "${it.subject}-${it.claimType}" })
                }
            }
        val findings =
            relationships.take(5).map {
                mapOf(
                    "summary" to "${it.sourceId} -> ${it.targetId}",
                    "explanation" to (it.description ?: it.type),
                )
            }
        val json =
            kotlinx.serialization.json.buildJsonObject {
                put("title", kotlinx.serialization.json.JsonPrimitive(title))
                put("summary", kotlinx.serialization.json.JsonPrimitive(summary))
                put("rating", kotlinx.serialization.json.JsonPrimitive(0))
                put("rating_explanation", kotlinx.serialization.json.JsonPrimitive("Fallback summary generated without LLM response."))
                put(
                    "findings",
                    kotlinx.serialization.json.JsonArray(
                        findings.map { finding ->
                            kotlinx.serialization.json.buildJsonObject {
                                put("summary", kotlinx.serialization.json.JsonPrimitive(finding["summary"] ?: ""))
                                put("explanation", kotlinx.serialization.json.JsonPrimitive(finding["explanation"] ?: ""))
                            }
                        },
                    ),
                )
            }
        return json.toString()
    }

    private fun runPrompt(
        inputText: String,
        maxLength: Int,
    ): String {
        val prompt =
            prompts
                .loadCommunityReportPrompt()
                .replace("{max_report_length}", maxLength.toString())
                .replace("{input_text}", inputText)
        return runCatching { reporter.chat(prompt) ?: "" }
            .getOrElse { e ->
                logger.warn(e) { "Community report prompt failed" }
                ""
            }.trim()
    }

    private interface Reporter {
        @dev.langchain4j.service.SystemMessage(
            "You are a helpful assistant that writes concise, well-grounded community reports based on supplied context.",
        )
        fun chat(
            @dev.langchain4j.service.UserMessage userMessage: String,
        ): String?
    }

    private fun textUnitsForCommunity(
        entities: List<Entity>,
        textUnits: List<TextUnit>,
    ): List<TextUnit> {
        if (entities.isEmpty() || textUnits.isEmpty()) return emptyList()
        val chunkIds = entities.map { it.sourceChunkId }.toSet()
        return textUnits.filter { it.chunkId in chunkIds }
    }

    private fun claimsForCommunity(
        entities: List<Entity>,
        claims: List<Claim>,
    ): List<Claim> {
        if (entities.isEmpty() || claims.isEmpty()) return emptyList()
        val names = entities.map { it.name.lowercase() }.toSet()
        return claims.filter { claim ->
            claim.subject.lowercase() in names || claim.`object`.lowercase() in names
        }
    }

    /**
     * Builds a CSV table of claims including a header row and one row per claim.
     *
     * Each field is escaped for CSV and rows are newline-separated. The header is:
     * "subject,object,type,status,start_date,end_date,description".
     *
     * @return The resulting CSV string with a header and one line per claim.
     */
    private fun buildClaimsTable(claims: List<Claim>): String =
        buildString {
            appendLine("subject,object,type,status,start_date,end_date,description")
            claims.forEach { c ->
                appendLine(
                    "${escapeCsv(c.subject)},${escapeCsv(c.`object`)},${escapeCsv(c.claimType)}," +
                        "${escapeCsv(c.status)},${escapeCsv(c.startDate)},${escapeCsv(c.endDate)}," +
                        escapeCsv(c.description),
                )
            }
        }

    /**
     * Escapes a string so it is safe to include as a CSV field.
     *
     * Replaces newline characters with spaces; if the result contains a comma or a quote,
     * wraps the value in double quotes and doubles any internal double quotes.
     *
     * @param value The input string to escape.
     * @return The escaped string suitable for CSV output.
     */
    private fun escapeCsv(value: String): String {
        val cleaned = value.replace("\n", " ")
        return if (cleaned.contains(",") || cleaned.contains("\"")) {
            "\"${cleaned.replace("\"", "\"\"")}\""
        } else {
            cleaned
        }
    }

    /**
     * Selects prior reports whose parentCommunityId matches the given community.
     *
     * @param communityId ID of the community to match.
     * @param priorReports List of prior community reports to filter.
     * @return A list of prior reports with `parentCommunityId` equal to the given `communityId`.
     */
    private fun subReportsForCommunity(
        communityId: Int,
        priorReports: List<CommunityReport>,
    ): List<CommunityReport> = priorReports.filter { it.parentCommunityId == communityId }
}
