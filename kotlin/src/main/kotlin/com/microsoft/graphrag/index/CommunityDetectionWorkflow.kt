package com.microsoft.graphrag.index

import nl.cwts.networkanalysis.LeidenAlgorithm
import nl.cwts.networkanalysis.Network
import nl.cwts.util.LargeDoubleArray
import nl.cwts.util.LargeIntArray
import java.util.Random

/**
 * Detects communities in an entity-relationship graph.
 */
@Suppress("LoopWithTooManyJumpStatements")
class CommunityDetectionWorkflow {
    /**
     * Runs community detection for the given entities and relationships.
     *
     * @param entities Entities to include as nodes.
     * @param relationships Relationships to include as edges.
     * @return CommunityDetectionResult containing assignments and hierarchy.
     */
    @Suppress("UnusedParameter")
    fun detect(
        entities: List<Entity>,
        relationships: List<Relationship>,
    ): CommunityDetectionResult {
        val graph = buildAdjacency(entities, relationships)
        val hierarchical = detectWithNetworkAnalysis(graph)
        if (hierarchical != null) {
            return hierarchical
        }
        val assignments = labelPropagation(graph)
        val communityAssignments =
            assignments.map { (entityId, community) -> CommunityAssignment(entityId, community) }
        return CommunityDetectionResult(communityAssignments, emptyMap())
    }

    private fun buildAdjacency(
        entities: List<Entity>,
        relationships: List<Relationship>,
    ): Map<String, MutableSet<String>> {
        val adjacency = mutableMapOf<String, MutableSet<String>>()
        entities.forEach { entity -> adjacency.computeIfAbsent(entity.id) { mutableSetOf() } }
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

    private fun detectWithNetworkAnalysis(graph: Map<String, MutableSet<String>>): CommunityDetectionResult? {
        return try {
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
            val clusterings =
                listOf(
                    LeidenAlgorithm(1.0, 10, 0.01, Random(40)).findClustering(network),
                    LeidenAlgorithm(0.5, 10, 0.01, Random(41)).findClustering(network),
                    LeidenAlgorithm(0.25, 10, 0.01, Random(42)).findClustering(network),
                )

            val indexToNode = nodeIndex.entries.associate { (node, idx) -> idx to node }
            val labelsByLevel = mutableListOf<Map<String, Int>>()
            clusterings.forEach { clustering ->
                val labels = mutableMapOf<String, Int>()
                indexToNode.forEach { (idx, nodeName) ->
                    labels[nodeName] = clustering.getCluster(idx)
                }
                labelsByLevel.add(normalizeLabels(labels))
            }

            val levelId: (Int, Int) -> Int = { level, cluster -> level * 1_000_000 + cluster }

            val fineLabels = labelsByLevel.firstOrNull() ?: emptyMap()
            val assignments =
                fineLabels.map { (node, community) ->
                    CommunityAssignment(node, levelId(0, community))
                }

            val hierarchy = mutableMapOf<Int, Int>()
            for (level in 0 until labelsByLevel.size - 1) {
                val childLabels = labelsByLevel[level]
                val parentLabels = labelsByLevel[level + 1]
                childLabels.forEach { (node, childCluster) ->
                    val parentCluster = parentLabels[node] ?: return@forEach
                    hierarchy[levelId(level, childCluster)] = levelId(level + 1, parentCluster)
                }
            }

            CommunityDetectionResult(assignments, hierarchy)
        } catch (_: Exception) {
            null
        }
    }

    private fun normalizeLabels(labels: Map<String, Int>): Map<String, Int> {
        val distinct = labels.values.distinct().sorted()
        val mapping = distinct.withIndex().associate { it.value to it.index }
        return labels.mapValues { (_, value) -> mapping[value] ?: value }
    }
}
