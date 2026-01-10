package com.microsoft.graphrag.index

import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.service.AiServices
import kotlinx.serialization.Serializable

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
        val tupleDelimiter = "<|>"
        val recordDelimiter = "##"
        val completionDelimiter = "<|COMPLETE|>"
        for (chunk in chunks) {
            val prompt =
                prompts
                    .loadExtractClaimsPrompt()
                    .replace("{entity_specs}", entitySpecs)
                    .replace("{claim_description}", claimDescription)
                    .replace("{tuple_delimiter}", tupleDelimiter)
                    .replace("{record_delimiter}", recordDelimiter)
                    .replace("{completion_delimiter}", completionDelimiter)
                    .replace("{input_text}", chunk.text)
            val content = extractor.chat(prompt)
            println("Claims raw response for chunk ${chunk.id}:\n$content")
            claims += parseClaims(content, tupleDelimiter, recordDelimiter, completionDelimiter)
        }
        return claims
    }

    private fun parseClaims(
        content: String,
        tupleDelimiter: String,
        recordDelimiter: String,
        completionDelimiter: String,
    ): List<Claim> {
        val cleaned =
            content
                .replace(completionDelimiter, "")
                .trim()
        val records =
            cleaned
                .split(recordDelimiter)
                .map { it.trim() }
                .filter { it.startsWith("(") && it.endsWith(")") }
        val claims = mutableListOf<Claim>()
        for (rec in records) {
            val body = rec.removePrefix("(").removeSuffix(")")
            val parts = body.split(tupleDelimiter)
            if (parts.size >= 8) {
                claims.add(
                    Claim(
                        subject = parts[0],
                        `object` = parts[1],
                        claimType = parts[2],
                        status = parts[3],
                        startDate = parts[4],
                        endDate = parts[5],
                        description = parts[6],
                        sourceText = parts[7],
                    ),
                )
            }
        }
        return claims
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
}
