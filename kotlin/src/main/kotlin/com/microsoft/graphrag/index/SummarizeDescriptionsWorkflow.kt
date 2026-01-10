package com.microsoft.graphrag.index

import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.openai.OpenAiChatModel
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

class SummarizeDescriptionsWorkflow(
    private val chatModel: OpenAiChatModel,
    private val prompts: PromptRepository = PromptRepository(),
    private val maxSummaryLength: Int = 120,
    private val maxInputTokens: Int = 800,
) {
    fun summarize(
        entities: List<Entity>,
        relationships: List<Relationship>,
        textUnits: List<TextUnit>,
    ): SummarizationResult {
        val entitySummaries =
            entities.map { entity ->
                val descriptions =
                    buildEntityDescriptions(
                        entity = entity,
                        relationships = relationships,
                        textUnits = textUnits,
                    )
                val summary = summarizeDescriptions(entity.name, descriptions)
                EntitySummary(entityId = entity.id, summary = summary)
            }

        val summarizedRelationships =
            relationships.map { relationship ->
                val descriptions =
                    buildRelationshipDescriptions(
                        relationship = relationship,
                    )
                val summary =
                    summarizeDescriptions(
                        idLabel = Json.encodeToString(listOf(relationship.sourceId, relationship.targetId)),
                        descriptions = descriptions,
                    )
                relationship.copy(description = summary.ifBlank { relationship.description ?: relationship.type })
            }

        return SummarizationResult(entitySummaries, summarizedRelationships)
    }

    private fun buildEntityDescriptions(
        entity: Entity,
        relationships: List<Relationship>,
        textUnits: List<TextUnit>,
    ): List<String> {
        val relatedDescriptions =
            relationships
                .filter { it.sourceId == entity.id || it.targetId == entity.id }
                .mapNotNull { it.description?.takeIf { desc -> desc.isNotBlank() } }
        val chunkContext =
            textUnits
                .filter { it.chunkId == entity.sourceChunkId }
                .map { "Context snippet: ${it.text.replace("\n", " ").take(400)}" }
        val fallback = listOf("${entity.type} entity named ${entity.name}")
        return (relatedDescriptions + chunkContext + fallback)
            .map { it.trim() }
            .filter { it.isNotEmpty() }
    }

    private fun buildRelationshipDescriptions(relationship: Relationship): List<String> {
        val desc = relationship.description?.takeIf { it.isNotBlank() }
        val fallback = "Relationship ${relationship.type} between ${relationship.sourceId} and ${relationship.targetId}"
        return listOfNotNull(desc, fallback)
    }

    private fun summarizeDescriptions(
        idLabel: String,
        descriptions: List<String>,
    ): String {
        if (descriptions.isEmpty()) return ""
        if (descriptions.size == 1) return descriptions.first()

        val promptTemplate = prompts.loadSummarizeDescriptionsPrompt()
        val promptTokenCost = tokenEstimate(promptTemplate)
        val maxTokensForContent = maxInputTokens - promptTokenCost

        var collected = mutableListOf<String>()
        var remainingTokens = maxTokensForContent
        var partialResults = descriptions.sorted()
        var result = ""

        partialResults.forEachIndexed { idx, description ->
            val descTokens = tokenEstimate(description)
            collected.add(description)
            remainingTokens -= descTokens

            val isLast = idx == partialResults.lastIndex
            val bufferFull = remainingTokens < 0 && collected.size > 1
            if (bufferFull || isLast) {
                result = callModel(idLabel, collected)
                if (!isLast) {
                    collected = mutableListOf(result)
                    remainingTokens = maxTokensForContent - tokenEstimate(result)
                }
            }
        }

        return result.ifBlank { descriptions.joinToString(" ") }
    }

    private fun callModel(
        idLabel: String,
        descriptions: List<String>,
    ): String {
        val prompt =
            prompts
                .loadSummarizeDescriptionsPrompt()
                .replace("{entity_name}", idLabel)
                .replace("{description_list}", Json.encodeToString(descriptions))
                .replace("{max_length}", maxSummaryLength.toString())
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

    private fun tokenEstimate(text: String): Int = (text.length / 4).coerceAtLeast(1)
}

data class SummarizationResult(
    val entitySummaries: List<EntitySummary>,
    val relationships: List<Relationship>,
)
