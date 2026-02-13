package com.microsoft.graphrag.logger

import io.github.oshai.kotlinlogging.KotlinLogging

data class Progress(
    val description: String? = null,
    val totalItems: Int? = null,
    val completedItems: Int? = null,
)

typealias ProgressHandler = (Progress) -> Unit

class ProgressTicker(
    private val callback: ProgressHandler?,
    private val numTotal: Int,
    private val description: String = "",
) {
    private var numComplete = 0

    /**
     * Advance the ticker by the given number of ticks and emit an updated progress snapshot.
     *
     * Builds a Progress containing the ticker's total and the new completed count, invokes the configured
     * callback with that Progress, and logs an informational message when a non-blank description is set.
     *
     * @param numTicks Number of completed items to advance; defaults to 1.
     */
    operator fun invoke(numTicks: Int = 1) {
        numComplete += numTicks
        val progress =
            Progress(
                totalItems = numTotal,
                completedItems = numComplete,
                description = description,
            )
        if (!progress.description.isNullOrBlank()) {
            logger.info { "${progress.description}${progress.completedItems}/${progress.totalItems}" }
        }
        callback?.invoke(progress)
    }

    /**
     * Marks the tracked work as complete and emits a final progress update.
     *
     * Invokes the configured callback with a Progress whose `completedItems` equals the total, and—if a
     * non-blank description is set—logs an informational message containing the description, completed and
     * total counts followed by " (done)".
     */
    fun done() {
        val progress =
            Progress(
                totalItems = numTotal,
                completedItems = numTotal,
                description = description,
            )
        if (!progress.description.isNullOrBlank()) {
            logger.info { "${progress.description}${progress.completedItems}/${progress.totalItems} (done)" }
        }
        callback?.invoke(progress)
    }

    companion object {
        private val logger = KotlinLogging.logger {}
    }
}

/**
 * Creates a ProgressTicker that reports progress for a known total.
 *
 * @param callback Optional handler invoked with each Progress update.
 * @param numTotal Total number of items to track.
 * @param description Optional text included with each Progress update; ignored if blank.
 * @return A configured ProgressTicker that emits progress updates when invoked.
 */
fun progressTicker(
    callback: ProgressHandler?,
    numTotal: Int,
    description: String = "",
): ProgressTicker = ProgressTicker(callback, numTotal, description)

/**
 * Wraps an Iterable to emit progress updates as its elements are consumed.
 *
 * The returned Iterable is lazy: each call to the iterator's `next()` advances an internal ticker
 * and invokes the provided `progress` callback with an updated Progress object.
 *
 * @param iterable The underlying sequence of elements to iterate.
 * @param progress Optional callback invoked with progress updates for each element consumed.
 * @param numTotal If provided, used as the total item count; otherwise falls back to collection size or errors.
 * @param description Optional human-readable description included in emitted Progress objects.
 * @return An Iterable that produces the same elements as `iterable` while reporting progress on consumption.
 */
fun <T> progressIterable(
    iterable: Iterable<T>,
    progress: ProgressHandler?,
    numTotal: Int? = null,
    description: String = "",
): Iterable<T> {
    val total =
        numTotal
            ?: (iterable as? Collection<*>)?.size
            ?: error("numTotal must be provided for non-Collection iterables")
    val ticker = progressTicker(progress, total, description)
    return Iterable {
        val iterator = iterable.iterator()
        object : Iterator<T> {
            override fun hasNext(): Boolean = iterator.hasNext()

            override fun next(): T {
                if (!iterator.hasNext()) throw NoSuchElementException()
                val value = iterator.next()
                ticker(1)
                return value
            }
        }
    }
}
