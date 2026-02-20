package com.microsoft.graphrag.evaluation

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class QAExactMatchTest {
    private val evaluator = QAExactMatch()

    @Test
    fun `calculateMetricScores should return 1 for perfect match`() {
        val goldAnswers = listOf(listOf("hello world"))
        val predictedAnswers = listOf("hello world")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        assertEquals(1.0, pooled["ExactMatch"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["ExactMatch"])
    }

    @Test
    fun `calculateMetricScores should return 0 for no match`() {
        val goldAnswers = listOf(listOf("hello world"))
        val predictedAnswers = listOf("goodbye world")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        assertEquals(0.0, pooled["ExactMatch"])
        assertEquals(1, perExample.size)
        assertEquals(0.0, perExample[0]["ExactMatch"])
    }

    @Test
    fun `calculateMetricScores should handle case insensitivity`() {
        val goldAnswers = listOf(listOf("Hello World"))
        val predictedAnswers = listOf("HELLO WORLD")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        assertEquals(1.0, pooled["ExactMatch"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["ExactMatch"])
    }

    @Test
    fun `calculateMetricScores should handle punctuation normalization`() {
        val goldAnswers = listOf(listOf("Hello, World!"))
        val predictedAnswers = listOf("Hello World")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        assertEquals(1.0, pooled["ExactMatch"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["ExactMatch"])
    }

    @Test
    fun `calculateMetricScores should handle article normalization`() {
        val goldAnswers = listOf(listOf("The quick fox"))
        val predictedAnswers = listOf("A quick fox")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // Both normalize to "quick fox"
        assertEquals(1.0, pooled["ExactMatch"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["ExactMatch"])
    }

    @Test
    fun `calculateMetricScores should handle whitespace normalization`() {
        val goldAnswers = listOf(listOf("hello   world"))
        val predictedAnswers = listOf("hello world")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        assertEquals(1.0, pooled["ExactMatch"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["ExactMatch"])
    }

    @Test
    fun `calculateMetricScores should handle empty predicted answer`() {
        val goldAnswers = listOf(listOf("hello world"))
        val predictedAnswers = listOf("")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        assertEquals(0.0, pooled["ExactMatch"])
        assertEquals(1, perExample.size)
        assertEquals(0.0, perExample[0]["ExactMatch"])
    }

    @Test
    fun `calculateMetricScores should handle empty gold answer`() {
        val goldAnswers = listOf(listOf(""))
        val predictedAnswers = listOf("")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // Both are empty, should match
        assertEquals(1.0, pooled["ExactMatch"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["ExactMatch"])
    }

    @Test
    fun `calculateMetricScores should handle multiple gold answers with max aggregation`() {
        val goldAnswers = listOf(listOf("cat", "dog", "pet"))
        val predictedAnswers = listOf("dog")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // Should match "dog" exactly
        assertEquals(1.0, pooled["ExactMatch"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["ExactMatch"])
    }

    @Test
    fun `calculateMetricScores should handle multiple gold answers with no match`() {
        val goldAnswers = listOf(listOf("cat", "dog", "pet"))
        val predictedAnswers = listOf("bird")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // None of the gold answers match
        assertEquals(0.0, pooled["ExactMatch"])
        assertEquals(1, perExample.size)
        assertEquals(0.0, perExample[0]["ExactMatch"])
    }

    @Test
    fun `calculateMetricScores should handle multiple gold answers with average aggregation`() {
        val goldAnswers = listOf(listOf("the cat", "the dog"))
        val predictedAnswers = listOf("cat")
        val aggregationFn: (List<Double>) -> Double = { if (it.isEmpty()) 0.0 else it.average() }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // "the cat" normalizes to "cat" (match=1.0), "the dog" normalizes to "dog" (match=0.0)
        // Average = 0.5
        assertEquals(0.5, pooled["ExactMatch"])
        assertEquals(1, perExample.size)
        assertEquals(0.5, perExample[0]["ExactMatch"])
    }

    @Test
    fun `calculateMetricScores should handle multiple examples`() {
        val goldAnswers = listOf(
            listOf("hello"),
            listOf("world"),
            listOf("test"),
        )
        val predictedAnswers = listOf(
            "hello",
            "world",
            "test",
        )
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        assertEquals(1.0, pooled["ExactMatch"])
        assertEquals(3, perExample.size)
        assertEquals(1.0, perExample[0]["ExactMatch"])
        assertEquals(1.0, perExample[1]["ExactMatch"])
        assertEquals(1.0, perExample[2]["ExactMatch"])
    }

    @Test
    fun `calculateMetricScores should compute average across multiple examples`() {
        val goldAnswers = listOf(
            listOf("match"),
            listOf("no match"),
            listOf("another match"),
            listOf("also no match"),
        )
        val predictedAnswers = listOf(
            "match",
            "different",
            "another match",
            "wrong",
        )
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // 2 matches out of 4 = 0.5
        assertEquals(0.5, pooled["ExactMatch"])
        assertEquals(4, perExample.size)
        assertEquals(1.0, perExample[0]["ExactMatch"])
        assertEquals(0.0, perExample[1]["ExactMatch"])
        assertEquals(1.0, perExample[2]["ExactMatch"])
        assertEquals(0.0, perExample[3]["ExactMatch"])
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
    fun `calculateMetricScores should handle empty gold answer list`() {
        val goldAnswers = listOf(emptyList<String>())
        val predictedAnswers = listOf("anything")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // Empty list means no possible matches
        assertEquals(0.0, pooled["ExactMatch"])
        assertEquals(1, perExample.size)
        assertEquals(0.0, perExample[0]["ExactMatch"])
    }

    @Test
    fun `calculateMetricScores should handle all empty inputs`() {
        val goldAnswers = listOf<List<String>>()
        val predictedAnswers = listOf<String>()
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        assertEquals(0.0, pooled["ExactMatch"])
        assertEquals(0, perExample.size)
    }

    @Test
    fun `calculateMetricScores should not be fooled by partial matches`() {
        val goldAnswers = listOf(listOf("hello world"))
        val predictedAnswers = listOf("hello")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // Partial match should be 0, not 1
        assertEquals(0.0, pooled["ExactMatch"])
        assertEquals(1, perExample.size)
        assertEquals(0.0, perExample[0]["ExactMatch"])
    }

    @Test
    fun `calculateMetricScores should handle numbers`() {
        val goldAnswers = listOf(listOf("The answer is 42"))
        val predictedAnswers = listOf("answer is 42")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // After normalization: "answer is 42" == "answer is 42"
        assertEquals(1.0, pooled["ExactMatch"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["ExactMatch"])
    }

    @Test
    fun `calculateMetricScores should handle complex normalization`() {
        val goldAnswers = listOf(listOf("The quick, brown fox!"))
        val predictedAnswers = listOf("a quick brown fox")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // Both normalize to "quick brown fox"
        assertEquals(1.0, pooled["ExactMatch"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["ExactMatch"])
    }

    @Test
    fun `calculateMetricScores should use min aggregation when specified`() {
        val goldAnswers = listOf(listOf("yes", "no"))
        val predictedAnswers = listOf("yes")
        val aggregationFn: (List<Double>) -> Double = { it.minOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // "yes" matches (1.0), "no" doesn't match (0.0), min = 0.0
        assertEquals(0.0, pooled["ExactMatch"])
        assertEquals(1, perExample.size)
        assertEquals(0.0, perExample[0]["ExactMatch"])
    }

    @Test
    fun `calculateMetricScores should handle unicode characters`() {
        val goldAnswers = listOf(listOf("café résumé"))
        val predictedAnswers = listOf("café résumé")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        assertEquals(1.0, pooled["ExactMatch"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["ExactMatch"])
    }

    @Test
    fun `calculateMetricScores should handle word order sensitivity`() {
        val goldAnswers = listOf(listOf("hello world"))
        val predictedAnswers = listOf("world hello")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // Word order matters for exact match
        assertEquals(0.0, pooled["ExactMatch"])
        assertEquals(1, perExample.size)
        assertEquals(0.0, perExample[0]["ExactMatch"])
    }

    @Test
    fun `calculateMetricScores should handle single character answers`() {
        val goldAnswers = listOf(listOf("a"))
        val predictedAnswers = listOf("a")
        val aggregationFn: (List<Double>) -> Double = { it.maxOrNull() ?: 0.0 }

        val (pooled, perExample) = evaluator.calculateMetricScores(goldAnswers, predictedAnswers, aggregationFn)

        // "a" is an article, so both normalize to empty string
        assertEquals(1.0, pooled["ExactMatch"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["ExactMatch"])
    }
}