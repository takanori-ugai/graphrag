package com.microsoft.graphrag.evaluation

/**
 * Token-level F1 evaluator for QA predictions.
 */
class QAF1Score {
    /**
     * Computes token-level F1 metrics for a set of question-answer predictions.
     *
     * @param goldAnswers Per-question list of valid gold answers; each element is a list of acceptable answer strings for that question.
     * @param predictedAnswers Predicted answer string for each question; must have the same length as `goldAnswers`.
     * @param aggregationFn Function used to aggregate multiple F1 scores when a question has multiple gold answers (receives the list of F1 scores and returns a single aggregated value).
     * @return A pair where the first element is a pooled metrics map containing a single entry `"F1"` with the average F1 across questions that have at least one gold answer, and the second element is a list of per-example metric maps (each map contains `"F1"` for that question).
     * Questions with an empty gold-answer list receive a per-example `"F1"` of `0.0` and are excluded from the pooled average.
     * @throws IllegalArgumentException if `goldAnswers` and `predictedAnswers` have different sizes.
     */
    fun calculateMetricScores(
        goldAnswers: List<List<String>>,
        predictedAnswers: List<String>,
        aggregationFn: (List<Double>) -> Double,
    ): Pair<Map<String, Double>, List<Map<String, Double>>> {
        require(goldAnswers.size == predictedAnswers.size) {
            "Length of gold answers and predicted answers should be the same."
        }

        /**
         * Computes the token-level F1 score between a gold answer and a predicted answer.
         *
         * The inputs are normalized and split on whitespace into tokens before computing overlap.
         *
         * @param gold The reference answer string.
         * @param predicted The predicted answer string.
         * @return The F1 score in the range 0.0 to 1.0: 1.0 indicates perfect token overlap, 0.0 indicates no overlap.
         */
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