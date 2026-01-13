package com.microsoft.graphrag.index

import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.service.AiServices
import io.github.oshai.kotlinlogging.KotlinLogging

class ClaimsWorkflow(
    private val chatModel: OpenAiChatModel,
    private val prompts: PromptRepository = PromptRepository(),
) {
    private val extractor =
        AiServices.create(Extractor::class.java, chatModel)

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
            val extracted =
                runCatching { extractor.extractClaims(prompt) }.getOrElse {
                    logger.warn { "Failed to parse claims JSON: ${it.message}" }
                    emptyList()
                }
            logger.debug { "Claims structured response for chunk ${chunk.id}: $extracted" }
            claims += extracted
        }
        return claims
    }

    private interface Extractor {
        @dev.langchain4j.service.SystemMessage(
            "You are an intelligent assistant that helps a human analyst to analyze " +
                "claims against certain entities presented in a text document.",
        )
        fun extractClaims(
            @dev.langchain4j.service.UserMessage userMessage: String,
        ): List<Claim>
    }

    companion object {
        private val logger = KotlinLogging.logger {}
    }
}
