package com.microsoft.graphrag.index

import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.service.AiServices
import java.util.UUID

class ExtractGraphWorkflow(
    private val chatModel: OpenAiChatModel,
    private val prompts: PromptRepository = PromptRepository(),
) {
    private val extractor =
        AiServices.create(Extractor::class.java, chatModel)

    suspend fun extract(chunks: List<DocumentChunk>): GraphExtractResult {
        val entities = mutableListOf<Entity>()
        val relationships = mutableListOf<Relationship>()

        for (chunk in chunks) {
            val prompt = buildPrompt(chunk)
            println("ExtractGraph: chunk ${chunk.id} text preview: ${chunk.text.take(200)}")
            val response = invokeChat(prompt)
            println("LLM raw response for chunk ${chunk.id}:\n$response")
            val parsed = parseResponse(response)
            val chunkEntities =
                parsed.entities.map {
                    it.copy(
                        id = it.id.ifBlank { UUID.randomUUID().toString() },
                        sourceChunkId = chunk.id,
                    )
                }
            val chunkRelationships =
                parsed.relationships.map {
                    it.copy(
                        sourceChunkId = chunk.id,
                    )
                }
            entities += chunkEntities
            relationships += chunkRelationships
        }

        return GraphExtractResult(entities = entities, relationships = relationships)
    }

    private fun buildPrompt(chunk: DocumentChunk): String =
        prompts
            .loadExtractGraphPrompt()
            .replace("{entity_types}", "ORGANIZATION,PERSON,GPE,LOCATION")
            .replace("{tuple_delimiter}", "|")
            .replace("{record_delimiter}", "\n")
            .replace("{completion_delimiter}", "__COMPLETE__")
            .replace("{input_text}", chunk.text)

    private fun parseResponse(content: String): GraphExtractResult {
        val entities = mutableListOf<Entity>()
        val relationships = mutableListOf<Relationship>()

        val entityRegex =
            Regex(
                """\("entity"\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\)""",
                setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL),
            )
        val relationshipRegex =
            Regex(
                """\("relationship"\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*([^)]+?)\)""",
                setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL),
            )

        entityRegex.findAll(content).forEach { m ->
            val name =
                m.groupValues
                    .getOrNull(1)
                    ?.trim()
                    ?.trim('"') ?: return@forEach
            val type =
                m.groupValues
                    .getOrNull(2)
                    ?.trim()
                    ?.trim('"') ?: return@forEach
            entities +=
                Entity(
                    id = UUID.randomUUID().toString(),
                    name = name,
                    type = type,
                    sourceChunkId = "",
                )
        }

        relationshipRegex.findAll(content).forEach { m ->
            val src =
                m.groupValues
                    .getOrNull(1)
                    ?.trim()
                    ?.trim('"') ?: return@forEach
            val tgt =
                m.groupValues
                    .getOrNull(2)
                    ?.trim()
                    ?.trim('"') ?: return@forEach
            val desc =
                m.groupValues
                    .getOrNull(3)
                    ?.trim()
                    ?.trim('"') ?: ""
            val strength =
                m.groupValues
                    .getOrNull(4)
                    ?.trim()
                    ?.trim('"') ?: ""
            relationships +=
                Relationship(
                    sourceId = src,
                    targetId = tgt,
                    type = desc.ifBlank { "related_to" },
                    description = "strength=$strength; $desc".trim(';', ' '),
                    sourceChunkId = "",
                )
        }

        return GraphExtractResult(entities, relationships)
    }

    private fun invokeChat(prompt: String): String {
        return extractor.chat(prompt)
    }

    private interface Extractor {
        @dev.langchain4j.service.SystemMessage(
            "You are an information extraction assistant. Extract entities and relationships exactly as instructed in the user message.",
        )
        fun chat(
            @dev.langchain4j.service.UserMessage userMessage: String,
        ): String
    }
}
