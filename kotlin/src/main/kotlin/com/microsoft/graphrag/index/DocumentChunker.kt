package com.microsoft.graphrag.index

import java.nio.file.Files
import java.nio.file.Path
import java.util.UUID

/**
 * Splits documents into overlapping chunks and converts them to text units.
 *
 * @property chunkSize Maximum size of each chunk in characters.
 * @property overlap Overlap size between consecutive chunks in characters.
 */
class DocumentChunker(
    private val chunkSize: Int = 1000,
    private val overlap: Int = 200,
) {
    /**
     * Loads all files under the input directory and chunks their content.
     *
     * @param inputDir Directory containing input documents.
     * @return List of document chunks across all files.
     */
    fun loadAndChunk(inputDir: Path): List<DocumentChunk> {
        if (!Files.exists(inputDir)) {
            return emptyList()
        }
        val chunks = mutableListOf<DocumentChunk>()
        Files.walk(inputDir).use { stream ->
            stream.filter { Files.isRegularFile(it) }.forEach { path ->
                val content = Files.readString(path)
                chunks += chunkText(content, path.toString())
            }
        }
        return chunks
    }

    /**
     * Chunks the provided text into overlapping slices.
     *
     * @param text Text to split into chunks.
     * @param sourcePath Source path recorded for each chunk.
     * @return List of generated document chunks.
     */
    fun chunkText(
        text: String,
        sourcePath: String,
    ): List<DocumentChunk> {
        val normalized = text.replace("\r\n", "\n")
        val result = mutableListOf<DocumentChunk>()
        var start = 0
        while (start < normalized.length) {
            val end = (start + chunkSize).coerceAtMost(normalized.length)
            val slice = normalized.substring(start, end)
            val id = UUID.randomUUID().toString()
            result.add(
                DocumentChunk(
                    id = id,
                    sourcePath = sourcePath,
                    text = slice,
                ),
            )
            if (end == normalized.length) {
                break
            }
            start += chunkSize - overlap
        }
        return result
    }

    /**
     * Converts document chunks into text units.
     *
     * @param chunks Chunks to convert.
     * @return List of text units derived from chunks.
     */
    fun toTextUnits(chunks: List<DocumentChunk>): List<TextUnit> =
        chunks.map { chunk ->
            TextUnit(
                id = "tu-${chunk.id}",
                chunkId = chunk.id,
                text = chunk.text,
                sourcePath = chunk.sourcePath,
            )
        }
}
