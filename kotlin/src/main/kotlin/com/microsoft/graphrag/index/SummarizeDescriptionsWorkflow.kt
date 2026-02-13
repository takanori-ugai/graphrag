package com.microsoft.graphrag.index

import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.service.AiServices
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

/**
 * Summarizes entity and relationship descriptions using an LLM.
 *
 * @property chatModel Chat model used to generate summaries.
 * @property prompts Repository that provides prompt templates.
 * @property maxSummaryLength Maximum summary length in characters.
 * @property maxInputTokens Token budget for input descriptions.
 */
class SummarizeDescriptionsWorkflow(
    private val chatModel: OpenAiChatModel,
    private val prompts: PromptRepository = PromptRepository(),
    private val maxSummaryLength: Int = 120,
    private val maxInputTokens: Int = 800,
) {
    private val summarizer = AiServices.create(Summarizer::class.java, chatModel)

    /**
     * Generates summaries for entities and relationships.
     *
     * @param entities Entities to summarize.
     * @param relationships Relationships to summarize.
     * @param textUnits Text units used for contextual snippets.
     * @return SummarizationResult containing entity summaries and updated relationships.
     */
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

    @Suppress("ReturnCount")
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
        val response = runCatching { summarizer.chat(prompt) }.getOrNull()
        return response?.trim().orEmpty()
    }

    private fun tokenEstimate(text: String): Int = (text.length / 4).coerceAtLeast(1)

    private interface Summarizer {
        @dev.langchain4j.service.SystemMessage(
            "You are an information extraction assistant. Summarize entity and relationship descriptions as instructed.",
        )
        fun chat(
            @dev.langchain4j.service.UserMessage userMessage: String,
        ): String?
    }
}

/**
 * Summary output for entities and relationships.
 *
 * @property entitySummaries Summaries for entities.
 * @property relationships Relationships with summarized descriptions.
 */
data class SummarizationResult(
    val entitySummaries: List<EntitySummary>,
    val relationships: List<Relationship>,
)
