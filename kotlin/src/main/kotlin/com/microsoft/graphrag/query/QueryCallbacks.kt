package com.microsoft.graphrag.query

/**
 * Callback hooks to mirror Python QueryCallbacks behaviour.
 */
interface QueryCallbacks {
    fun onContext(context: Map<String, List<MutableMap<String, String>>>) {}

    fun onLLMNewToken(token: String) {}

    fun onMapResponseStart(mapContexts: List<String>) {}

    fun onMapResponseEnd(mapResponses: List<QueryResult>) {}

    fun onReduceResponseStart(reduceContext: String) {}

    fun onReduceResponseEnd(reduceResponse: String) {}
}
