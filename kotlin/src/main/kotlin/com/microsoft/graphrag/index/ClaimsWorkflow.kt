package com.microsoft.graphrag.index

import com.fasterxml.jackson.databind.ObjectMapper
import dev.langchain4j.data.message.ChatMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.openai.OpenAiChatModel
import io.github.oshai.kotlinlogging.KotlinLogging

class ClaimsWorkflow(
    private val chatModel: OpenAiChatModel,
    private val prompts: PromptRepository = PromptRepository(),
    private val objectMapper: ObjectMapper = ObjectMapper().findAndRegisterModules(),
) {
    fun extractClaims(
        chunks: List<DocumentChunk>,
        entitySpecs: String = "ORGANIZATION,PERSON,GPE",
        claimDescription: String = "Any claims or facts that could be relevant to information discovery.",
    ): List<Claim> {
        val claims = mutableListOf<Claim>()
        val basePrompt = prompts.loadExtractClaimsPrompt()
        for (chunk in chunks) {
            val prompt =
                basePrompt
                    .replace("{entity_specs}", entitySpecs)
                    .replace("{claim_description}", claimDescription)
                    .replace("{input_text}", chunk.text)
            val extracted =
                runCatching {
                    extractClaimsJson(prompt)
                }.getOrElse {
                    logger.warn { "Failed to parse claims JSON: ${it.message}" }
                    emptyList()
                }
            logger.debug { "Claims parsed for chunk ${chunk.id}: ${extracted.size}" }
            claims += extracted
        }
        return claims
    }

    private fun extractClaimsJson(prompt: String): List<Claim> {
        val messages =
            listOf<ChatMessage>(
                UserMessage(prompt),
            )
        val response = chat(messages)
        if (response.isBlank()) return emptyList()

        val json = extractJsonArray(response) ?: return emptyList()
        return runCatching {
            objectMapper.readValue(json, Array<Claim>::class.java).toList()
        }.getOrElse {
            logger.warn { "Claims JSON decode failed: ${it.message}" }
            emptyList()
        }
    }

    private fun extractJsonArray(text: String): String? {
        val start = text.indexOf('[')
        val end = text.lastIndexOf(']')
        if (start == -1 || end == -1 || end <= start) return null
        return text.substring(start, end + 1)
    }

    private fun chat(messages: List<ChatMessage>): String =
        runCatching { chatModel.chat(messages).aiMessage().text() }
            .getOrElse {
                logger.warn { "Claim extraction chat failed: ${it.message}" }
                ""
            }

    companion object {
        private val logger = KotlinLogging.logger {}
    }
}
