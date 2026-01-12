package com.microsoft.graphrag.query

/**
 * Captures callback events so streaming code paths can still surface context and response metadata.
 */
class CollectingQueryCallbacks : QueryCallbacks {
    var contextRecords: Map<String, List<MutableMap<String, String>>> = emptyMap()
        private set
    var mapResponses: List<QueryResult> = emptyList()
        private set
    var reduceContext: String = ""
        private set
    var reduceResponse: String = ""
        private set

    override fun onContext(context: Map<String, List<MutableMap<String, String>>>) {
        contextRecords = context
    }

    override fun onMapResponseEnd(mapResponses: List<QueryResult>) {
        this.mapResponses = mapResponses
    }

    override fun onReduceResponseStart(reduceContext: String) {
        this.reduceContext = reduceContext
    }

    override fun onReduceResponseEnd(reduceResponse: String) {
        this.reduceResponse = reduceResponse
    }
}
