package com.microsoft.graphrag.index

import com.microsoft.graphrag.logger.ProgressHandler
import com.microsoft.graphrag.logger.progressTicker
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.openai.OpenAiEmbeddingModel
import dev.langchain4j.model.output.Response
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class EmbedWorkflow(
    private val embeddingModel: EmbeddingModel,
) {
    /**
         * Produce vector embeddings for the given document chunks using the workflow's embedding model.
         *
         * Processes each chunk in an IO dispatcher, reporting progress via the optional handler and
         * associating each successfully computed vector with its chunk id.
         *
         * @param chunks The document chunks to embed.
         * @param progress Optional progress handler that receives periodic updates during embedding.
         * @param description Description passed to the progress ticker for display or logging.
         * @return A list of `TextEmbedding` objects for chunks whose embeddings were successfully computed; chunks that fail to embed are omitted. 
         */
        suspend fun embedChunks(
        chunks: List<DocumentChunk>,
        progress: ProgressHandler? = null,
        description: String = "embed chunks progress: ",
    ): List<TextEmbedding> =
        withContext(Dispatchers.IO) {
            val ticker = progressTicker(progress, chunks.size, description)
            val results =
                chunks.mapNotNull { chunk ->
                    val vector = embed(chunk.text) ?: return@mapNotNull null
                    ticker()
                    TextEmbedding(chunkId = chunk.id, vector = vector)
                }
            ticker.done()
            results
        }

    /**
         * Produces vector embeddings for the provided entities' names.
         *
         * @param entities The entities to embed; each entity's `name` is used to compute its embedding.
         * @param progress Optional progress handler called as each embedding completes.
         * @param description Human-readable description forwarded to the progress ticker.
         * @return A list of EntityEmbedding objects pairing `entityId` with the computed embedding vector; entities whose embeddings fail are omitted.
         */
        suspend fun embedEntities(
        entities: List<Entity>,
        progress: ProgressHandler? = null,
        description: String = "embed entities progress: ",
    ): List<EntityEmbedding> =
        withContext(Dispatchers.IO) {
            val ticker = progressTicker(progress, entities.size, description)
            val results =
                entities.mapNotNull { entity ->
                    val vector = embed(entity.name) ?: return@mapNotNull null
                    ticker()
                    EntityEmbedding(entityId = entity.id, vector = vector)
                }
            ticker.done()
            results
        }

    /**
         * Computes the embedding vector for the given text.
         *
         * @param text The input text to embed.
         * @return `List<Double>` containing the embedding vector, or `null` if an error occurs while obtaining the embedding.
         */
        private fun embed(text: String): List<Double>? =
        try {
            val response: Response<dev.langchain4j.data.embedding.Embedding> = embeddingModel.embed(text)
            response
                .content()
                ?.vector()
                ?.asList()
                ?.map { it.toDouble() }
        } catch (e: Exception) {
            logger.warn(e) { "Embedding failed for text: ${text.take(50)}..." }
            null
        }

    companion object {
        private val logger = KotlinLogging.logger {}
    }
}

/**
         * Creates an OpenAI-backed EmbeddingModel configured with the given API key and model name.
         *
         * @param apiKey The OpenAI API key used to authenticate requests.
         * @param modelName The embedding model identifier to use (default: "text-embedding-3-small").
         * @return An EmbeddingModel instance that sends embedding requests to the specified OpenAI model.
         */
        fun defaultEmbeddingModel(
    apiKey: String,
    modelName: String = "text-embedding-3-small",
): EmbeddingModel =
    OpenAiEmbeddingModel
        .builder()
        .apiKey(apiKey)
        .modelName(modelName)
        .build()