package com.microsoft.graphrag.index

import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.openai.OpenAiChatModel

class CommunityReportWorkflow(
    private val chatModel: OpenAiChatModel,
) {
    fun generateReports(
        assignments: List<CommunityAssignment>,
        entities: List<Entity>,
    ): List<CommunityReport> {
        val byCommunity = assignments.groupBy { it.communityId }
        return byCommunity.map { (communityId, members) ->
            val names =
                members.mapNotNull { assignment ->
                    entities.find { it.id == assignment.entityId }?.name
                }
            val summary =
                if (names.isEmpty()) {
                    "No entities in this community."
                } else {
                    summarizeCommunity(communityId, names)
                }
            CommunityReport(communityId = communityId, summary = summary)
        }
    }

    private fun summarizeCommunity(
        communityId: Int,
        names: List<String>,
    ): String {
        val prompt =
            """
            Summarize the following group of entities (community $communityId) in 3-5 bullet points.
            Entities: ${names.joinToString(", ")}
            """.trimIndent()
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
}
