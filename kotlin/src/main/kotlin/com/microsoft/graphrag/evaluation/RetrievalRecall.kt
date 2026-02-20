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
     */
    fun calculateMetricScores(
        goldDocs: List<List<String>>,
        retrievedDocs: List<List<String>>,
        kList: List<Int>,
    ): Pair<Map<String, Double>, List<Map<String, Double>>> {
        val uniqueKList = kList.distinct().sorted()
        val exampleEvalResults = mutableListOf<Map<String, Double>>()
        val pooledEvalResults = uniqueKList.associate { k -> "Recall@$k" to 0.0 }.toMutableMap()

        for ((exampleGoldDocs, exampleRetrievedDocs) in goldDocs.zip(retrievedDocs)) {
            val maxK = uniqueKList.last()
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
