package com.microsoft.graphrag.index

import java.nio.file.Files
import java.nio.file.Path
import java.util.UUID

class DocumentChunker(
    private val chunkSize: Int = 1000,
    private val overlap: Int = 200,
) {
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
            start = end - overlap
            if (start < 0) start = 0
            if (start >= normalized.length) break
        }
        return result
    }
}
