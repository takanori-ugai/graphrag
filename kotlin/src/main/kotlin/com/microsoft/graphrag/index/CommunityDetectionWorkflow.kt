package com.microsoft.graphrag.index

import nl.cwts.networkanalysis.LeidenAlgorithm
import nl.cwts.networkanalysis.Network
import nl.cwts.util.LargeDoubleArray
import nl.cwts.util.LargeIntArray
import java.util.Random

@Suppress("LoopWithTooManyJumpStatements")
class CommunityDetectionWorkflow {
    @Suppress("UnusedParameter")
    fun detect(
        entities: List<Entity>,
        relationships: List<Relationship>,
    ): List<CommunityAssignment> {
        val graph = buildAdjacency(relationships)
        val networkAssignments = detectWithNetworkAnalysis(graph)
        val assignments = networkAssignments ?: labelPropagation(graph)
        return assignments.map { (entityId, community) -> CommunityAssignment(entityId, community) }
    }

    private fun buildAdjacency(relationships: List<Relationship>): Map<String, MutableSet<String>> {
        val adjacency = mutableMapOf<String, MutableSet<String>>()
        relationships.forEach { rel ->
            adjacency.computeIfAbsent(rel.sourceId) { mutableSetOf() }.add(rel.targetId)
            adjacency.computeIfAbsent(rel.targetId) { mutableSetOf() }.add(rel.sourceId)
        }
        return adjacency
    }

    // Simple label propagation as a fallback community detector.
    private fun labelPropagation(
        graph: Map<String, MutableSet<String>>,
        maxIters: Int = 10,
    ): Map<String, Int> {
        val labels = mutableMapOf<String, Int>()
        var labelCounter = 0
        graph.keys.forEach { node -> labels[node] = labelCounter++ }

        var iterations = 0
        while (iterations < maxIters) {
            var changed = false
            for ((node, neighbors) in graph) {
                if (neighbors.isEmpty()) continue
                val neighborLabels = neighbors.mapNotNull { labels[it] }
                if (neighborLabels.isEmpty()) continue
                val mode =
                    neighborLabels
                        .groupingBy { it }
                        .eachCount()
                        .maxByOrNull { it.value }
                        ?.key
                if (mode != null && mode != labels[node]) {
                    labels[node] = mode
                    changed = true
                }
            }
            iterations++
            if (!changed) {
                break
            }
        }

        // normalize labels to consecutive ints
        val mapping =
            labels.values
                .distinct()
                .sorted()
                .withIndex()
                .associate { it.value to it.index }
        return labels.mapValues { (_, label) -> mapping[label] ?: label }
    }

    private fun detectWithNetworkAnalysis(graph: Map<String, MutableSet<String>>): Map<String, Int>? =
        try {
            val nodeIndex = graph.keys.withIndex().associate { it.value to it.index }
            if (nodeIndex.isEmpty()) return null

            val neighborLists = Array(nodeIndex.size) { mutableListOf<Int>() }
            graph.forEach { (node, neighbors) ->
                val src = nodeIndex[node] ?: return@forEach
                neighbors.forEach { neighbor ->
                    val tgt = nodeIndex[neighbor] ?: return@forEach
                    neighborLists[src].add(tgt)
                }
            }

            val edges: Array<LargeIntArray> =
                Array(neighborLists.size) { idx ->
                    LargeIntArray(neighborLists[idx].toIntArray())
                }
            val edgeWeights = LargeDoubleArray(neighborLists.sumOf { it.size }.toLong())
            val nodeWeights = DoubleArray(nodeIndex.size) { 1.0 }

            val network = Network(nodeWeights, edges, edgeWeights, false, true)
            val leiden = LeidenAlgorithm(1.0, 10, 0.01, Random(42))
            val clustering = leiden.findClustering(network)

            val result = mutableMapOf<String, Int>()
            val indexToNode = nodeIndex.entries.associate { (node, idx) -> idx to node }
            indexToNode.forEach { (idx, nodeName) ->
                result[nodeName] = clustering.getCluster(idx)
            }
            result
        } catch (_: Exception) {
            null
        }
}
