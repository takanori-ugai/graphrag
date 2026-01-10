package com.microsoft.graphrag.index

import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.openai.OpenAiChatModel

class SummarizeDescriptionsWorkflow(
    private val chatModel: OpenAiChatModel,
    private val prompts: PromptRepository = PromptRepository(),
) {
    fun summarize(entities: List<Entity>): List<EntitySummary> {
        val results = mutableListOf<EntitySummary>()
        entities.forEach { entity ->
            val descriptionList = listOf("${entity.type} entity named ${entity.name}")
            val prompt =
                prompts
                    .loadSummarizeDescriptionsPrompt()
                    .replace("{entity_name}", entity.name)
                    .replace("{description_list}", descriptionList.joinToString("; "))
                    .replace("{max_length}", "100")
            val message = UserMessage.from(prompt)
            val method =
                chatModel.javaClass.methods.firstOrNull { it.name == "generate" && it.parameterTypes.size == 1 }
            val result = method?.invoke(chatModel, listOf(message))
            val summary =
                when (result) {
                    is dev.langchain4j.data.message.AiMessage -> result.text()
                    is dev.langchain4j.model.output.Response<*> -> result.content().toString()
                    else -> result?.toString() ?: ""
                }
            results.add(EntitySummary(entityId = entity.id, summary = summary))
        }
        return results
    }
}
