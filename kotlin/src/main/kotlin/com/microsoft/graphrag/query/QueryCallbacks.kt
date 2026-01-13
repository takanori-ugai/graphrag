package com.microsoft.graphrag.query

/**
 * Callback hooks to mirror Python QueryCallbacks behaviour.
 */
interface QueryCallbacks {
    /**
     * Invoked when context information for a query is available.
     *
     * @param context Map from context identifier to a list of mutable key/value maps representing context entries.
     */
    fun onContext(context: Map<String, List<MutableMap<String, String>>>) {}

    /**
     * Invoked when the language model produces a new token.
     *
     * @param token The token text emitted by the LLM.
     */
    fun onLLMNewToken(token: String) {}

    /**
     * Invoked at the start of a map response.
     *
     * @param mapContexts Identifiers of the map contexts involved in the response.
     */
    fun onMapResponseStart(mapContexts: List<String>) {}

    /**
     * Called when a map response finishes, delivering the collected map-phase results.
     *
     * @param mapResponses The list of map responses produced during the map phase as `QueryResult` objects.
     */
    fun onMapResponseEnd(mapResponses: List<QueryResult>) {}

    /**
     * Invoked when a reduce-stage response begins.
     *
     * @param reduceContext The reduce response context string (for example, the reduce prompt or context identifier).
     */
    fun onReduceResponseStart(reduceContext: String) {}

    /**
     * Invoked when the reduce-phase response completes.
     *
     * @param reduceResponse The final aggregated response produced by the reduce step.
     */
    fun onReduceResponseEnd(reduceResponse: String) {}
}
