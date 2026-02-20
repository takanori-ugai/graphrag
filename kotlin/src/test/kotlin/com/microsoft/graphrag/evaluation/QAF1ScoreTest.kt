package com.microsoft.graphrag.evaluation

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class QAF1ScoreTest {
    private val evaluator = QAF1Score()

    @Test
    fun `calculateMetricScores should compute F1 for perfect match`() {
        val goldAnswers = listOf(listOf("the quick brown fox"))
        val predictedAnswers = listOf("the quick brown fox")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        assertEquals(1.0, pooled["F1"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["F1"])
    }

    @Test
    fun `calculateMetricScores should compute F1 for partial match`() {
        val goldAnswers = listOf(listOf("the quick brown fox"))
        val predictedAnswers = listOf("the quick brown dog")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // "quick brown" are common (2 tokens), fox vs dog differ
        // After normalization: "quick brown fox" vs "quick brown dog"
        // Precision = 2/3, Recall = 2/3, F1 = 2/3
        assertTrue(pooled["F1"]!! > 0.0)
        assertTrue(pooled["F1"]!! < 1.0)
        assertEquals(1, perExample.size)
    }

    @Test
    fun `calculateMetricScores should compute F1 for no match`() {
        val goldAnswers = listOf(listOf("apple"))
        val predictedAnswers = listOf("orange")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        assertEquals(0.0, pooled["F1"])
        assertEquals(1, perExample.size)
        assertEquals(0.0, perExample[0]["F1"])
    }

    @Test
    fun `calculateMetricScores should handle empty predicted answer`() {
        val goldAnswers = listOf(listOf("hello world"))
        val predictedAnswers = listOf("")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        assertEquals(0.0, pooled["F1"])
        assertEquals(1, perExample.size)
        assertEquals(0.0, perExample[0]["F1"])
    }

    @Test
    fun `calculateMetricScores should handle empty gold answer`() {
        val goldAnswers = listOf(listOf(""))
        val predictedAnswers = listOf("hello world")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        assertEquals(0.0, pooled["F1"])
        assertEquals(1, perExample.size)
        assertEquals(0.0, perExample[0]["F1"])
    }

    @Test
    fun `calculateMetricScores should handle empty gold answer list`() {
        val goldAnswers = listOf(emptyList<String>())
        val predictedAnswers = listOf("hello world")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // Empty gold list should give F1=0.0 but not count toward average
        assertEquals(0.0, pooled["F1"])
        assertEquals(1, perExample.size)
        assertEquals(0.0, perExample[0]["F1"])
    }

    @Test
    fun `calculateMetricScores should handle multiple gold answers using max aggregation`() {
        val goldAnswers = listOf(listOf("cat", "dog", "pet"))
        val predictedAnswers = listOf("dog")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // Should match "dog" perfectly (F1=1.0)
        assertEquals(1.0, pooled["F1"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["F1"])
    }

    @Test
    fun `calculateMetricScores should handle multiple gold answers using average aggregation`() {
        val goldAnswers = listOf(listOf("the cat", "the dog"))
        val predictedAnswers = listOf("cat")
        // Average aggregation
        val aggregationFn: (List<Double>) -> Double = { if (it.isEmpty()) 0.0 else it.average() }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // F1 against "the cat" = 1.0 (after normalization both are "cat")
        // F1 against "the dog" = 0.0 (no overlap)
        // Average = 0.5
        assertEquals(0.5, pooled["F1"])
        assertEquals(1, perExample.size)
        assertEquals(0.5, perExample[0]["F1"])
    }

    @Test
    fun `calculateMetricScores should handle multiple examples`() {
        val goldAnswers = listOf(
            listOf("hello world"),
            listOf("goodbye world"),
            listOf("hello friend"),
        )
        val predictedAnswers = listOf(
            "hello world",
            "goodbye world",
            "hello friend",
        )
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        assertEquals(1.0, pooled["F1"])
        assertEquals(3, perExample.size)
        assertEquals(1.0, perExample[0]["F1"])
        assertEquals(1.0, perExample[1]["F1"])
        assertEquals(1.0, perExample[2]["F1"])
    }

    @Test
    fun `calculateMetricScores should compute average F1 across multiple examples`() {
        val goldAnswers = listOf(
            listOf("perfect match"),
            listOf("no match at all"),
        )
        val predictedAnswers = listOf(
            "perfect match",
            "completely different",
        )
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // First example: F1=1.0, Second example: F1=0.0
        // Average = 0.5
        assertEquals(0.5, pooled["F1"])
        assertEquals(2, perExample.size)
    }

    @Test
    fun `calculateMetricScores should throw exception for mismatched sizes`() {
        val goldAnswers = listOf(listOf("answer1"), listOf("answer2"))
        val predictedAnswers = listOf("prediction1")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        assertFailsWith<IllegalArgumentException> {
            evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)
        }
    }

    @Test
    fun `calculateMetricScores should handle case insensitivity via normalization`() {
        val goldAnswers = listOf(listOf("Hello World"))
        val predictedAnswers = listOf("HELLO WORLD")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        assertEquals(1.0, pooled["F1"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["F1"])
    }

    @Test
    fun `calculateMetricScores should handle punctuation via normalization`() {
        val goldAnswers = listOf(listOf("Hello, World!"))
        val predictedAnswers = listOf("Hello World")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        assertEquals(1.0, pooled["F1"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["F1"])
    }

    @Test
    fun `calculateMetricScores should handle articles via normalization`() {
        val goldAnswers = listOf(listOf("The quick fox"))
        val predictedAnswers = listOf("A quick fox")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // Both normalize to "quick fox"
        assertEquals(1.0, pooled["F1"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["F1"])
    }

    @Test
    fun `calculateMetricScores should handle token repetition in predicted answer`() {
        val goldAnswers = listOf(listOf("hello world"))
        val predictedAnswers = listOf("hello hello world world")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // Gold: ["hello", "world"] (2 tokens)
        // Predicted: ["hello", "hello", "world", "world"] (4 tokens)
        // Common: min counts = hello(1), world(1) = 2 tokens
        // Precision = 2/4 = 0.5, Recall = 2/2 = 1.0
        // F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 1.0 / 1.5 = 0.6667
        assertTrue(pooled["F1"]!! > 0.6)
        assertTrue(pooled["F1"]!! < 0.7)
    }

    @Test
    fun `calculateMetricScores should handle token repetition in gold answer`() {
        val goldAnswers = listOf(listOf("hello hello world world"))
        val predictedAnswers = listOf("hello world")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // Gold: ["hello", "hello", "world", "world"] (4 tokens)
        // Predicted: ["hello", "world"] (2 tokens)
        // Common: min counts = hello(1), world(1) = 2 tokens
        // Precision = 2/2 = 1.0, Recall = 2/4 = 0.5
        // F1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 1.0 / 1.5 = 0.6667
        assertTrue(pooled["F1"]!! > 0.6)
        assertTrue(pooled["F1"]!! < 0.7)
    }

    @Test
    fun `calculateMetricScores should exclude empty gold lists from pooled average`() {
        val goldAnswers = listOf(
            emptyList(),
            listOf("hello world"),
        )
        val predictedAnswers = listOf(
            "anything",
            "hello world",
        )
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // First example has empty gold list, so it gets F1=0.0 but excluded from average
        // Second example has F1=1.0
        // Pooled average should be 1.0 (only counting valid examples)
        assertEquals(1.0, pooled["F1"])
        assertEquals(2, perExample.size)
        assertEquals(0.0, perExample[0]["F1"])
        assertEquals(1.0, perExample[1]["F1"])
    }

    @Test
    fun `calculateMetricScores should handle all empty gold lists`() {
        val goldAnswers = listOf(emptyList<String>(), emptyList())
        val predictedAnswers = listOf("answer1", "answer2")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // All examples excluded from average, should be 0.0
        assertEquals(0.0, pooled["F1"])
        assertEquals(2, perExample.size)
    }

    @Test
    fun `calculateMetricScores should handle single token match`() {
        val goldAnswers = listOf(listOf("yes"))
        val predictedAnswers = listOf("yes")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        assertEquals(1.0, pooled["F1"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["F1"])
    }

    @Test
    fun `calculateMetricScores should compute F1 with min aggregation for multiple golds`() {
        val goldAnswers = listOf(listOf("cat", "feline animal"))
        val predictedAnswers = listOf("cat")
        // Min aggregation - worst case
        val aggregationFn: (List<Double>) -> Double = { it.minOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // F1 against "cat" = 1.0
        // F1 against "feline animal" = something less than 1.0 (no overlap)
        // Min should give us the lower value
        assertTrue(pooled["F1"]!! < 1.0)
        assertEquals(1, perExample.size)
    }
}