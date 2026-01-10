package com.microsoft.graphrag.index

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.file.Files
import java.nio.file.Path
import java.util.stream.Collectors

interface PipelineStorage {
    suspend fun get(name: String): String?

    suspend fun set(
        name: String,
        content: String,
    )

    fun child(name: String): PipelineStorage

    fun find(regex: Regex): Sequence<Pair<String, Path>>
}

class FilePipelineStorage(
    private val root: Path,
) : PipelineStorage {
    override suspend fun get(name: String): String? =
        withContext(Dispatchers.IO) {
            val target = root.resolve(name)
            if (!Files.exists(target)) {
                return@withContext null
            }
            Files.readString(target)
        }

    override suspend fun set(
        name: String,
        content: String,
    ) {
        withContext(Dispatchers.IO) {
            val target = root.resolve(name)
            if (!Files.exists(target.parent)) {
                Files.createDirectories(target.parent)
            }
            Files.writeString(target, content)
        }
    }

    override fun child(name: String): PipelineStorage = FilePipelineStorage(root.resolve(name))

    override fun find(regex: Regex): Sequence<Pair<String, Path>> {
        if (!Files.exists(root)) {
            return emptySequence()
        }
        return Files.walk(root).use { stream ->
            stream
                .filter { Files.isRegularFile(it) }
                .map { path ->
                    val relative = root.relativize(path).toString()
                    relative to path
                }.filter { (relative, _) -> regex.containsMatchIn(relative) }
                .collect(Collectors.toList())
                .asSequence()
        }
    }
}
