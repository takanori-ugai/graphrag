package com.microsoft.graphrag.index

import org.jgrapht.graph.DefaultDirectedGraph
import org.jgrapht.graph.DefaultEdge

/**
 * Builds directed graphs from entity and relationship records.
 */
class GraphBuilder {
    /**
     * Constructs a directed graph with entity ids as vertices and relationships as edges.
     *
     * @param entities Entities to add as vertices.
     * @param relationships Relationships to add as directed edges.
     * @return A directed graph populated with entity vertices and relationship edges.
     */
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
