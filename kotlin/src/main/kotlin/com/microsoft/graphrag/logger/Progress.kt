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

    fun done() {
        callback?.invoke(
            Progress(
                totalItems = numTotal,
                completedItems = numTotal,
                description = description,
            ),
        )
    }

    companion object {
        private val logger = KotlinLogging.logger {}
    }
}

fun progressTicker(
    callback: ProgressHandler?,
    numTotal: Int,
    description: String = "",
): ProgressTicker = ProgressTicker(callback, numTotal, description)

fun <T> progressIterable(
    iterable: Iterable<T>,
    progress: ProgressHandler?,
    numTotal: Int? = null,
    description: String = "",
): Iterable<T> {
    val total = numTotal ?: iterable.count()
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
