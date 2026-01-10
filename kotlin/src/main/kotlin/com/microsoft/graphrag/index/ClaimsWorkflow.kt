package com.microsoft.graphrag.index

import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.openai.OpenAiChatModel
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json

class ClaimsWorkflow(
    private val chatModel: OpenAiChatModel,
    private val prompts: PromptRepository = PromptRepository(),
) {
    fun extractClaims(
        chunks: List<DocumentChunk>,
        entitySpecs: String = "ORGANIZATION,PERSON,GPE",
        claimDescription: String = "red flags associated with an entity",
    ): List<Claim> {
        val claims = mutableListOf<Claim>()
        for (chunk in chunks) {
            val prompt =
                prompts
                    .loadExtractClaimsPrompt()
                    .replace("{entity_specs}", entitySpecs)
                    .replace("{claim_description}", claimDescription)
                    .replace("{tuple_delimiter}", "|")
                    .replace("{record_delimiter}", "\n")
                    .replace("{completion_delimiter}", "__COMPLETE__")
                    .replace("{input_text}", chunk.text)
            val message = UserMessage.from(prompt)
            val method =
                chatModel.javaClass.methods.firstOrNull { it.name == "generate" && it.parameterTypes.size == 1 }
            val result = method?.invoke(chatModel, listOf(message))
            val content =
                when (result) {
                    is dev.langchain4j.data.message.AiMessage -> result.text()
                    is dev.langchain4j.model.output.Response<*> -> result.content().toString()
                    else -> result?.toString() ?: ""
                }
            claims += parseClaims(content)
        }
        return claims
    }

    private fun parseClaims(content: String): List<Claim> {
        // naive parse: split by record delimiter and tuple delimiter
        val records = content.split("\n").map { it.trim() }.filter { it.startsWith("(") && it.endsWith(")") }
        val claims = mutableListOf<Claim>()
        for (rec in records) {
            val body = rec.removePrefix("(").removeSuffix(")")
            val parts = body.split("|")
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
}
