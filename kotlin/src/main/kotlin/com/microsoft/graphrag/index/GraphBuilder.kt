package com.microsoft.graphrag.index

import org.jgrapht.graph.DefaultDirectedGraph
import org.jgrapht.graph.DefaultEdge

class GraphBuilder {
    fun buildGraph(
        entities: List<Entity>,
        relationships: List<Relationship>,
    ): DefaultDirectedGraph<String, DefaultEdge> {
        val graph = DefaultDirectedGraph<String, DefaultEdge>(DefaultEdge::class.java)
        entities.forEach { graph.addVertex(it.id) }
        relationships.forEach { rel ->
            if (!graph.containsVertex(rel.sourceId)) graph.addVertex(rel.sourceId)
            if (!graph.containsVertex(rel.targetId)) graph.addVertex(rel.targetId)
            graph.addEdge(rel.sourceId, rel.targetId)
        }
        return graph
    }
}
