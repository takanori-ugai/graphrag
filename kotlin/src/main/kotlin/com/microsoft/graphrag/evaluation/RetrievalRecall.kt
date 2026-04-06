package com.microsoft.graphrag.evaluation

import io.github.oshai.kotlinlogging.KotlinLogging
import java.util.Locale

/**
 * Recall evaluator for retrieval results at multiple k values.
 */
class RetrievalRecall {
    private val logger = KotlinLogging.logger {}

    /**
     * Calculates recall@k metrics for each example and pooled averages.
     *
     * @param goldDocs List of gold document ids, one per example
     * @param retrievedDocs List of retrieved document ids, one per example
     * @param kList List of k values for recall@k computation
     * @return Pair of (1) pooled metrics across all examples and (2) per-example metric maps
     * If kList is empty, returns empty pooled and per-example results.
     * Examples with empty gold doc lists receive per-example recall=0.0 and are included in the pooled average.
     * @throws IllegalArgumentException if goldDocs and retrievedDocs have different sizes
     */
    fun calculateMetricScores(
        goldDocs: List<List<String>>,
        retrievedDocs: List<List<String>>,
        kList: List<Int>,
    ): Pair<Map<String, Double>, List<Map<String, Double>>> {
        require(goldDocs.size == retrievedDocs.size) {
            "Length of gold docs and retrieved docs must be the same."
        }
        val uniqueKList = kList.distinct().sorted()
        if (uniqueKList.isEmpty()) {
            return emptyMap<String, Double>() to emptyList()
        }
        val exampleEvalResults = mutableListOf<Map<String, Double>>()
        val pooledEvalResults = uniqueKList.associate { k -> "Recall@$k" to 0.0 }.toMutableMap()
        val maxK = uniqueKList.last()

        for ((exampleGoldDocs, exampleRetrievedDocs) in goldDocs.zip(retrievedDocs)) {
            if (exampleRetrievedDocs.size < maxK) {
                logger.warn {
                    "Length of retrieved docs (${exampleRetrievedDocs.size}) is smaller than largest topk for recall score ($maxK)"
                }
            }

            val exampleEvalResult = uniqueKList.associate { k -> "Recall@$k" to 0.0 }.toMutableMap()
            val goldDocSet = exampleGoldDocs.toSet()

            for (k in uniqueKList) {
                val topKDocs = exampleRetrievedDocs.take(k)
                val relevantRetrieved = topKDocs.toSet().intersect(goldDocSet)
                val recall =
                    if (goldDocSet.isNotEmpty()) {
                        relevantRetrieved.size.toDouble() / goldDocSet.size.toDouble()
                    } else {
                        0.0
                    }
                exampleEvalResult["Recall@$k"] = recall
            }

            exampleEvalResults.add(exampleEvalResult)

            for (k in uniqueKList) {
                val key = "Recall@$k"
                pooledEvalResults[key] = (pooledEvalResults[key] ?: 0.0) + (exampleEvalResult[key] ?: 0.0)
            }
        }

        val numExamples = goldDocs.size
        if (numExamples > 0) {
            for (k in uniqueKList) {
                val key = "Recall@$k"
                pooledEvalResults[key] = (pooledEvalResults[key] ?: 0.0) / numExamples.toDouble()
            }
        }

        val roundedPooled = pooledEvalResults.mapValues { (_, v) -> String.format(Locale.US, "%.4f", v).toDouble() }
        return roundedPooled to exampleEvalResults
    }
}
