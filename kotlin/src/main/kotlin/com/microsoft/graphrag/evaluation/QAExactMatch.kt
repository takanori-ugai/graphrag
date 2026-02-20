package com.microsoft.graphrag.evaluation

/**
 * Exact match evaluator for QA predictions.
 */
class QAExactMatch {
    /**
     * Calculates exact match metrics for the provided answers.
     */
    fun calculateMetricScores(
        goldAnswers: List<List<String>>,
        predictedAnswers: List<String>,
        aggregationFn: (List<Double>) -> Double,
    ): Pair<Map<String, Double>, List<Map<String, Double>>> {
        require(goldAnswers.size == predictedAnswers.size) {
            "Length of gold answers and predicted answers should be the same."
        }

        val exampleEvalResults = mutableListOf<Map<String, Double>>()
        var totalEm = 0.0

        for ((goldList, predicted) in goldAnswers.zip(predictedAnswers)) {
            val normalizedPredicted = normalizeAnswer(predicted)
            val emScores =
                goldList.map { gold ->
                    if (normalizeAnswer(gold) == normalizedPredicted) 1.0 else 0.0
                }
            val aggregatedEm = aggregationFn(emScores)
            exampleEvalResults.add(mapOf("ExactMatch" to aggregatedEm))
            totalEm += aggregatedEm
        }

        val avgEm = if (goldAnswers.isNotEmpty()) totalEm / goldAnswers.size else 0.0
        val pooledEvalResults = mapOf("ExactMatch" to avgEm)

        return pooledEvalResults to exampleEvalResults
    }
}
