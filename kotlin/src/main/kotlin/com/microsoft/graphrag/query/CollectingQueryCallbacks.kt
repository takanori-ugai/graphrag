package com.microsoft.graphrag.query

/**
 * Captures callback events so streaming code paths can still surface context and response metadata.
 */
class CollectingQueryCallbacks : QueryCallbacks {
    /**
     * Captured context records converted to immutable maps.
     */
    var contextRecords: Map<String, List<Map<String, String>>> = emptyMap()
        private set

    /**
     * Map-phase responses captured from callback events.
     */
    var mapResponses: List<QueryResult> = emptyList()
        private set

    /**
     * Reduce-phase context string captured when reduce starts.
     */
    var reduceContext: String = ""
        private set

    /**
     * Final reduce-phase response captured when reduce ends.
     */
    var reduceResponse: String = ""
        private set

    /**
     * Stores a snapshot of streaming context records, converting each mutable record to an immutable map.
     *
     * @param context A map from context key to a list of mutable record maps; each record is converted to an
     * immutable `Map<String, String>` and stored in `contextRecords`.
     */
    override fun onContext(context: Map<String, List<MutableMap<String, String>>>) {
        contextRecords = context.mapValues { (_, records) -> records.map { it.toMap() } }
    }

    /**
     * Creates a mutable copy of the stored context records.
     *
     * @return A map with the same keys as `contextRecords` where each value is a list of mutable maps
     *         containing the same key/value pairs as the original records.
     */
    fun toMutableContextRecords(): Map<String, List<MutableMap<String, String>>> =
        contextRecords.mapValues { (_, records) -> records.map { it.toMutableMap() } }

    /**
     * Stores the completed map-phase responses into this callback's state.
     *
     * @param mapResponses The list of `QueryResult` objects produced by the map phase to record.
     */
    override fun onMapResponseEnd(mapResponses: List<QueryResult>) {
        this.mapResponses = mapResponses
    }

    /**
     * Stores the provided reduce-phase context string for later retrieval.
     *
     * @param reduceContext The context string emitted when a reduce response starts.
     */
    override fun onReduceResponseStart(reduceContext: String) {
        this.reduceContext = reduceContext
    }

    /**
     * Captures and stores the final reduce-phase response string.
     *
     * @param reduceResponse The reduce response text to record.
     */
    override fun onReduceResponseEnd(reduceResponse: String) {
        this.reduceResponse = reduceResponse
    }
}
