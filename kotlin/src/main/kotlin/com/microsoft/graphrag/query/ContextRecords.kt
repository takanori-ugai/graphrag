package com.microsoft.graphrag.query

internal fun Map<String, List<MutableMap<String, String>>>.toImmutableContextRecords(): Map<String, List<Map<String, String>>> =
    mapValues { (_, records) -> records.map { it.toMap() } }

internal fun Map<String, List<Map<String, String>>>.toMutableContextRecords(): Map<String, List<MutableMap<String, String>>> =
    mapValues { (_, records) -> records.map { it.toMutableMap() } }
