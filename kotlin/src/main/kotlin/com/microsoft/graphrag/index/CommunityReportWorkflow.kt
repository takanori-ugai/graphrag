package com.microsoft.graphrag.index

import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.openai.OpenAiChatModel

@Suppress("UnusedParameter")
class CommunityReportWorkflow(
    private val chatModel: OpenAiChatModel,
    private val prompts: PromptRepository = PromptRepository(),
) {
    fun generateReports(
        assignments: List<CommunityAssignment>,
        entities: List<Entity>,
        relationships: List<Relationship>,
    ): List<CommunityReport> {
        val byCommunity = assignments.groupBy { it.communityId }
        return byCommunity.map { (communityId, members) ->
            val communityEntities = entities.filter { e -> members.any { it.entityId == e.id } }
            val communityRelationships =
                relationships.filter { rel ->
                    members.any { it.entityId == rel.sourceId || it.entityId == rel.targetId }
                }
            val summary =
                if (communityEntities.isEmpty()) {
                    "No entities in this community."
                } else {
                    summarizeCommunity(communityId, communityEntities, communityRelationships)
                }
            CommunityReport(communityId = communityId, summary = summary)
        }
    }

    private fun summarizeCommunity(
        communityId: Int,
        entities: List<Entity>,
        relationships: List<Relationship>,
    ): String {
        val entitiesTable = buildEntitiesTable(entities)
        val relationshipsTable = buildRelationshipsTable(relationships)
        val inputText = "Entities\n$entitiesTable\n\nRelationships\n$relationshipsTable"
        val prompt =
            prompts
                .loadCommunityReportPrompt()
                .replace("{max_report_length}", "300")
                .replace("{input_text}", inputText)
        val message = UserMessage.from(prompt)
        val method =
            chatModel.javaClass.methods.firstOrNull { it.name == "generate" && it.parameterTypes.size == 1 }
        val result = method?.invoke(chatModel, listOf(message))
        return when (result) {
            is dev.langchain4j.data.message.AiMessage -> result.text()
            is dev.langchain4j.model.output.Response<*> -> result.content().toString()
            else -> result?.toString() ?: ""
        }
    }

    private fun buildEntitiesTable(entities: List<Entity>): String =
        buildString {
            appendLine("id,entity,description")
            entities.forEachIndexed { idx, e ->
                appendLine("${idx + 1},${e.name},${e.type}: ${e.name}")
            }
        }

    private fun buildRelationshipsTable(relationships: List<Relationship>): String =
        buildString {
            appendLine("id,source,target,description")
            relationships.forEachIndexed { idx, r ->
                appendLine("${idx + 1},${r.sourceId},${r.targetId},${r.description ?: r.type}")
            }
        }
}
