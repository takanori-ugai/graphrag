package com.microsoft.graphrag.evaluation

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class RetrievalRecallTest {
    private val evaluator = RetrievalRecall()

    @Test
    fun `calculateMetricScores should return perfect recall when all gold docs are retrieved`() {
        val goldDocs = listOf(listOf("doc1", "doc2", "doc3"))
        val retrievedDocs = listOf(listOf("doc1", "doc2", "doc3"))
        val kList = listOf(3)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        assertEquals(1.0, pooled["Recall@3"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["Recall@3"])
    }

    @Test
    fun `calculateMetricScores should return zero recall when no gold docs are retrieved`() {
        val goldDocs = listOf(listOf("doc1", "doc2", "doc3"))
        val retrievedDocs = listOf(listOf("doc4", "doc5", "doc6"))
        val kList = listOf(3)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        assertEquals(0.0, pooled["Recall@3"])
        assertEquals(1, perExample.size)
        assertEquals(0.0, perExample[0]["Recall@3"])
    }

    @Test
    fun `calculateMetricScores should compute partial recall`() {
        val goldDocs = listOf(listOf("doc1", "doc2", "doc3", "doc4"))
        val retrievedDocs = listOf(listOf("doc1", "doc2", "doc5", "doc6"))
        val kList = listOf(4)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        // 2 out of 4 gold docs retrieved = 0.5
        assertEquals(0.5, pooled["Recall@4"])
        assertEquals(1, perExample.size)
        assertEquals(0.5, perExample[0]["Recall@4"])
    }

    @Test
    fun `calculateMetricScores should handle multiple k values`() {
        val goldDocs = listOf(listOf("doc1", "doc2", "doc3"))
        val retrievedDocs = listOf(listOf("doc1", "doc5", "doc2", "doc6", "doc3"))
        val kList = listOf(1, 3, 5)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        // At k=1: only doc1 retrieved, recall = 1/3 ≈ 0.3333
        assertEquals(0.3333, pooled["Recall@1"])
        // At k=3: doc1 and doc2 retrieved, recall = 2/3 ≈ 0.6667
        assertEquals(0.6667, pooled["Recall@3"])
        // At k=5: all three retrieved, recall = 3/3 = 1.0
        assertEquals(1.0, pooled["Recall@5"])

        assertEquals(1, perExample.size)
        assertEquals(1.0 / 3.0, perExample[0]["Recall@1"]!!, 1e-6)
        assertEquals(2.0 / 3.0, perExample[0]["Recall@3"]!!, 1e-6)
        assertEquals(1.0, perExample[0]["Recall@5"])
    }

    @Test
    fun `calculateMetricScores should handle empty gold docs`() {
        val goldDocs = listOf(emptyList<String>())
        val retrievedDocs = listOf(listOf("doc1", "doc2", "doc3"))
        val kList = listOf(3)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        // No gold docs means recall is 0.0
        assertEquals(0.0, pooled["Recall@3"])
        assertEquals(1, perExample.size)
        assertEquals(0.0, perExample[0]["Recall@3"])
    }

    @Test
    fun `calculateMetricScores should handle empty retrieved docs`() {
        val goldDocs = listOf(listOf("doc1", "doc2", "doc3"))
        val retrievedDocs = listOf(emptyList<String>())
        val kList = listOf(3)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        // No retrieved docs means recall is 0.0
        assertEquals(0.0, pooled["Recall@3"])
        assertEquals(1, perExample.size)
        assertEquals(0.0, perExample[0]["Recall@3"])
    }

    @Test
    fun `calculateMetricScores should handle retrieved docs smaller than k`() {
        val goldDocs = listOf(listOf("doc1", "doc2", "doc3"))
        val retrievedDocs = listOf(listOf("doc1", "doc2"))
        val kList = listOf(5)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        // Only 2 docs retrieved, can check at most 2, recall = 2/3 ≈ 0.6667
        assertEquals(0.6667, pooled["Recall@5"])
        assertEquals(1, perExample.size)
        assertEquals(2.0 / 3.0, perExample[0]["Recall@5"]!!, 1e-6)
    }

    @Test
    fun `calculateMetricScores should handle multiple examples`() {
        val goldDocs =
            listOf(
                listOf("doc1", "doc2"),
                listOf("doc3", "doc4"),
                listOf("doc5", "doc6"),
            )
        val retrievedDocs =
            listOf(
                listOf("doc1", "doc2", "doc7"),
                listOf("doc3", "doc8", "doc9"),
                listOf("doc10", "doc11", "doc12"),
            )
        val kList = listOf(3)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        // Example 1: 2/2 = 1.0
        // Example 2: 1/2 = 0.5
        // Example 3: 0/2 = 0.0
        // Average = 0.5
        assertEquals(0.5, pooled["Recall@3"])
        assertEquals(3, perExample.size)
        assertEquals(1.0, perExample[0]["Recall@3"])
        assertEquals(0.5, perExample[1]["Recall@3"])
        assertEquals(0.0, perExample[2]["Recall@3"])
    }

    @Test
    fun `calculateMetricScores should deduplicate k values`() {
        val goldDocs = listOf(listOf("doc1", "doc2"))
        val retrievedDocs = listOf(listOf("doc1", "doc2"))
        val kList = listOf(2, 2, 2, 2)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        // Should only have one entry for Recall@2
        assertEquals(1, pooled.size)
        assertEquals(1.0, pooled["Recall@2"])
        assertEquals(1, perExample.size)
        assertEquals(1, perExample[0].size)
    }

    @Test
    fun `calculateMetricScores should sort k values`() {
        val goldDocs = listOf(listOf("doc1", "doc2", "doc3"))
        val retrievedDocs = listOf(listOf("doc1", "doc5", "doc2", "doc6", "doc3"))
        val kList = listOf(5, 1, 3)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        // Should have results for k=1, k=3, k=5
        assertTrue(pooled.containsKey("Recall@1"))
        assertTrue(pooled.containsKey("Recall@3"))
        assertTrue(pooled.containsKey("Recall@5"))
    }

    @Test
    fun `calculateMetricScores should handle single gold doc`() {
        val goldDocs = listOf(listOf("doc1"))
        val retrievedDocs = listOf(listOf("doc1", "doc2", "doc3"))
        val kList = listOf(3)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        assertEquals(1.0, pooled["Recall@3"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["Recall@3"])
    }

    @Test
    fun `calculateMetricScores should handle single retrieved doc`() {
        val goldDocs = listOf(listOf("doc1", "doc2", "doc3"))
        val retrievedDocs = listOf(listOf("doc1"))
        val kList = listOf(1)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        assertEquals(0.3333, pooled["Recall@1"])
        assertEquals(1, perExample.size)
        assertEquals(1.0 / 3.0, perExample[0]["Recall@1"]!!, 1e-6)
    }

    @Test
    fun `calculateMetricScores should handle duplicate docs in gold set`() {
        val goldDocs = listOf(listOf("doc1", "doc1", "doc2"))
        val retrievedDocs = listOf(listOf("doc1", "doc2", "doc3"))
        val kList = listOf(3)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        // Gold set is converted to set, so {"doc1", "doc2"} (2 unique)
        // Both are retrieved, so recall = 2/2 = 1.0
        assertEquals(1.0, pooled["Recall@3"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["Recall@3"])
    }

    @Test
    fun `calculateMetricScores should handle duplicate docs in retrieved list`() {
        val goldDocs = listOf(listOf("doc1", "doc2"))
        val retrievedDocs = listOf(listOf("doc1", "doc1", "doc2"))
        val kList = listOf(3)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        // Retrieved set is converted to set for intersection
        // Both gold docs are in retrieved set, recall = 2/2 = 1.0
        assertEquals(1.0, pooled["Recall@3"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["Recall@3"])
    }

    @Test
    fun `calculateMetricScores should respect document order with k`() {
        val goldDocs = listOf(listOf("doc1", "doc2", "doc3"))
        val retrievedDocs = listOf(listOf("doc4", "doc5", "doc1", "doc2", "doc3"))
        val kList = listOf(2)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        // At k=2, only doc4 and doc5 are considered, neither is in gold set
        assertEquals(0.0, pooled["Recall@2"])
        assertEquals(1, perExample.size)
        assertEquals(0.0, perExample[0]["Recall@2"])
    }

    @Test
    fun `calculateMetricScores should round pooled results to 4 decimal places`() {
        val goldDocs = listOf(listOf("doc1", "doc2", "doc3"))
        val retrievedDocs = listOf(listOf("doc1", "doc4", "doc5"))
        val kList = listOf(3)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        // 1/3 = 0.333333... should be rounded to 0.3333
        val recall = pooled["Recall@3"]!!
        assertEquals(0.3333, recall)
        // Ensure it's actually rounded to 4 decimal places
        assertEquals("0.3333", recall.toString())
    }

    @Test
    fun `calculateMetricScores should handle no examples`() {
        val goldDocs = emptyList<List<String>>()
        val retrievedDocs = emptyList<List<String>>()
        val kList = listOf(3)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        assertEquals(0.0, pooled["Recall@3"])
        assertEquals(0, perExample.size)
    }

    @Test
    fun `calculateMetricScores should compute recall at k=1 correctly`() {
        val goldDocs = listOf(listOf("doc1", "doc2", "doc3"))
        val retrievedDocs = listOf(listOf("doc1"))
        val kList = listOf(1)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        // Top 1 doc is doc1, which is in gold set, so 1/3 ≈ 0.3333
        assertEquals(0.3333, pooled["Recall@1"])
        assertEquals(1, perExample.size)
        assertEquals(1.0 / 3.0, perExample[0]["Recall@1"]!!, 1e-6)
    }

    @Test
    fun `calculateMetricScores should handle large k values`() {
        val goldDocs = listOf(listOf("doc1", "doc2"))
        val retrievedDocs = listOf(listOf("doc1", "doc2", "doc3"))
        val kList = listOf(100)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        // Even though k=100, only 3 docs retrieved, both gold docs are in there
        assertEquals(1.0, pooled["Recall@100"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["Recall@100"])
    }

    @Test
    fun `calculateMetricScores should handle mixed results across examples`() {
        val goldDocs =
            listOf(
                listOf("doc1"),
                listOf("doc2"),
                listOf("doc3"),
                listOf("doc4"),
            )
        val retrievedDocs =
            listOf(
                listOf("doc1"),
                listOf("doc5"),
                listOf("doc3"),
                listOf("doc6"),
            )
        val kList = listOf(1)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        // 2 matches out of 4 = 0.5
        assertEquals(0.5, pooled["Recall@1"])
        assertEquals(4, perExample.size)
        assertEquals(1.0, perExample[0]["Recall@1"])
        assertEquals(0.0, perExample[1]["Recall@1"])
        assertEquals(1.0, perExample[2]["Recall@1"])
        assertEquals(0.0, perExample[3]["Recall@1"])
    }

    @Test
    fun `calculateMetricScores should handle recall improving with larger k`() {
        val goldDocs = listOf(listOf("doc1", "doc2", "doc3", "doc4", "doc5"))
        val retrievedDocs = listOf(listOf("doc6", "doc1", "doc7", "doc2", "doc8", "doc3", "doc4", "doc5"))
        val kList = listOf(2, 4, 6, 8)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        // At k=2: only doc6, doc1 -> 1/5 = 0.2
        // At k=4: doc6, doc1, doc7, doc2 -> 2/5 = 0.4
        // At k=6: doc6, doc1, doc7, doc2, doc8, doc3 -> 3/5 = 0.6
        // At k=8: all -> 5/5 = 1.0
        assertEquals(0.2, pooled["Recall@2"])
        assertEquals(0.4, pooled["Recall@4"])
        assertEquals(0.6, pooled["Recall@6"])
        assertEquals(1.0, pooled["Recall@8"])
    }

    @Test
    fun `calculateMetricScores should handle case sensitivity correctly`() {
        val goldDocs = listOf(listOf("Doc1", "Doc2"))
        val retrievedDocs = listOf(listOf("doc1", "doc2"))
        val kList = listOf(2)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        // String comparison is case-sensitive, so no matches
        assertEquals(0.0, pooled["Recall@2"])
        assertEquals(1, perExample.size)
        assertEquals(0.0, perExample[0]["Recall@2"])
    }

    @Test
    fun `calculateMetricScores should handle whitespace in doc IDs`() {
        val goldDocs = listOf(listOf("doc 1", "doc 2"))
        val retrievedDocs = listOf(listOf("doc 1", "doc 2"))
        val kList = listOf(2)

        val (pooled, perExample) = evaluator.calculateMetricScores(goldDocs, retrievedDocs, kList)

        assertEquals(1.0, pooled["Recall@2"])
        assertEquals(1, perExample.size)
        assertEquals(1.0, perExample[0]["Recall@2"])
    }
}
