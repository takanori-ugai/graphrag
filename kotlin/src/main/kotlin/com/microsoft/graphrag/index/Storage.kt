package com.microsoft.graphrag.index

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.file.Files
import java.nio.file.Path
import java.util.stream.Collectors

/**
 * Abstraction for pipeline artifact storage.
 */
interface PipelineStorage {
    /**
     * Reads a stored value by name.
     *
     * @param name Relative name or key of the stored value.
     * @return The stored content, or null if missing.
     */
    suspend fun get(name: String): String?

    /**
     * Writes content to the given name.
     *
     * @param name Relative name or key to write.
     * @param content Content to store.
     */
    suspend fun set(
        name: String,
        content: String,
    )

    /**
     * Returns a storage instance rooted at a child path.
     *
     * @param name Child directory name.
     * @return Storage rooted at the child path.
     */
    fun child(name: String): PipelineStorage

    /**
     * Finds files whose relative paths match the provided regex.
     *
     * @param regex Pattern to match relative paths against.
     * @return Sequence of relative path and absolute path pairs.
     */
    fun find(regex: Regex): Sequence<Pair<String, Path>>
}

/**
 * Filesystem-backed implementation of PipelineStorage.
 *
 * @property root Root directory for stored artifacts.
 */
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
