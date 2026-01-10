package com.microsoft.graphrag.index

import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.service.AiServices
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json

class ClaimsWorkflow(
    private val chatModel: OpenAiChatModel,
    private val prompts: PromptRepository = PromptRepository(),
) {
    fun extractClaims(
        chunks: List<DocumentChunk>,
        entitySpecs: String = "ORGANIZATION,PERSON,GPE",
        claimDescription: String = "Any claims or facts that could be relevant to information discovery.",
    ): List<Claim> {
        val claims = mutableListOf<Claim>()
        for (chunk in chunks) {
            val prompt =
                prompts
                    .loadExtractClaimsPrompt()
                    .replace("{entity_specs}", entitySpecs)
                    .replace("{claim_description}", claimDescription)
                    .replace("{input_text}", chunk.text)
            val content = extractor.chat(prompt)
            logger.debug { "Claims raw response for chunk ${chunk.id}:\n$content" }
            claims += parseClaims(content)
        }
        return claims
    }

    private fun parseClaims(content: String): List<Claim> {
        val cleaned =
            content
                .trim()
                .removePrefix("```json")
                .removePrefix("```")
                .removeSuffix("```")
                .trim()
        val jsonText = cleaned.substringAfter("[", cleaned).substringBeforeLast("]", cleaned).let { "[$it]" }
        val decoded =
            runCatching {
                Json.decodeFromString(ListSerializer(Claim.serializer()), jsonText)
            }.getOrElse {
                logger.warn { "Failed to parse claims JSON: ${it.message}" }
                emptyList()
            }
        if (decoded.isNotEmpty()) {
            return decoded
        }
        return emptyList()
    }

    private val extractor =
        AiServices.create(Extractor::class.java, chatModel)

    private interface Extractor {
        @dev.langchain4j.service.SystemMessage(
            "You are an intelligent assistant that helps a human analyst to analyze claims against certain entities presented in a text document.",
        )
        fun chat(
            @dev.langchain4j.service.UserMessage userMessage: String,
        ): String
    }

    companion object {
        private val logger = KotlinLogging.logger {}
    }
}
