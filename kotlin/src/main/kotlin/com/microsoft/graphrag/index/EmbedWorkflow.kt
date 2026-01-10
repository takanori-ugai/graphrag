package com.microsoft.graphrag.index

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.openai.OpenAiEmbeddingModel
import dev.langchain4j.model.output.Response
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class EmbedWorkflow(
    private val embeddingModel: EmbeddingModel,
) {
    suspend fun embedChunks(chunks: List<DocumentChunk>): List<TextEmbedding> =
        withContext(Dispatchers.IO) {
            chunks.mapNotNull { chunk ->
                val vector = embed(chunk.text) ?: return@mapNotNull null
                TextEmbedding(chunkId = chunk.id, vector = vector)
            }
        }

    suspend fun embedEntities(entities: List<Entity>): List<EntityEmbedding> =
        withContext(Dispatchers.IO) {
            entities.mapNotNull { entity ->
                val vector = embed(entity.name) ?: return@mapNotNull null
                EntityEmbedding(entityId = entity.id, vector = vector)
            }
        }

    private fun embed(text: String): List<Double>? {
        val method =
            embeddingModel.javaClass.methods.firstOrNull { it.name == "embed" && it.parameterTypes.size == 1 }
        val response = method?.invoke(embeddingModel, text)
        return when (response) {
            is Response<*> -> {
                val content = response.content()
                if (content is Embedding) {
                    content.vector().asList().map { it.toDouble() }
                } else {
                    null
                }
            }

            else -> {
                null
            }
        }
    }
}

fun defaultEmbeddingModel(apiKey: String): EmbeddingModel =
    OpenAiEmbeddingModel
        .builder()
        .apiKey(apiKey)
        .modelName("text-embedding-3-small")
        .build()
