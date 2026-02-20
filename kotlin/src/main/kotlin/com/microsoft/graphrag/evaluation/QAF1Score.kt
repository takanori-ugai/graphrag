package com.microsoft.graphrag.evaluation


/**
 * Token-level F1 evaluator for QA predictions.
 */
class QAF1Score {
    /**
     * Calculates F1 metrics for the provided answers.
     *
     * @param goldAnswers List of gold answer sets, one per question (each question may have multiple valid answers)
     * @param predictedAnswers List of predicted answers, one per question
     * @param aggregationFn Function to aggregate multiple F1 scores when multiple gold answers exist
     * @return Pair of (1) pooled metrics across all examples and (2) per-example metric maps
     * Questions with empty gold answer lists receive per-example F1=0.0 but are excluded from the pooled average.
     * @throws IllegalArgumentException if goldAnswers and predictedAnswers have different sizes
     */
    fun calculateMetricScores(
        goldAnswers: List<List<String>>,
        predictedAnswers: List<String>,
        aggregationFn: (List<Double>) -> Double,
    ): Pair<Map<String, Double>, List<Map<String, Double>>> {
        require(goldAnswers.size == predictedAnswers.size) {
            "Length of gold answers and predicted answers should be the same."
        }

        fun computeF1(
            gold: String,
            predicted: String,
        ): Double {
            val goldTokens = normalizeAnswer(gold).split(" ").filter { it.isNotEmpty() }
            val predictedTokens = normalizeAnswer(predicted).split(" ").filter { it.isNotEmpty() }
            if (goldTokens.isEmpty() || predictedTokens.isEmpty()) return 0.0

            val goldCounts = goldTokens.groupingBy { it }.eachCount()
            val predictedCounts = predictedTokens.groupingBy { it }.eachCount()

            var numSame = 0
            for ((token, count) in predictedCounts) {
                val goldCount = goldCounts[token] ?: 0
                numSame += minOf(count, goldCount)
            }

            if (numSame == 0) return 0.0

            val precision = numSame.toDouble() / predictedTokens.size.toDouble()
            val recall = numSame.toDouble() / goldTokens.size.toDouble()
            return 2 * (precision * recall) / (precision + recall)
        }

        val exampleEvalResults = mutableListOf<Map<String, Double>>()
        var totalF1 = 0.0
        var validCount = 0

        for ((goldList, predicted) in goldAnswers.zip(predictedAnswers)) {
            if (goldList.isEmpty()) {
                exampleEvalResults.add(mapOf("F1" to 0.0))
                continue
            }
            val f1Scores = goldList.map { gold -> computeF1(gold, predicted) }
            val aggregatedF1 = aggregationFn(f1Scores)
            exampleEvalResults.add(mapOf("F1" to aggregatedF1))
            totalF1 += aggregatedF1
            validCount++
        }

        val avgF1 = if (validCount > 0) totalF1 / validCount else 0.0
        val pooledEvalResults = mapOf("F1" to avgF1)

        return pooledEvalResults to exampleEvalResults
    }
}
