package com.microsoft.graphrag.query

/**
 * Produces an immutable representation of context records keyed by string.
 *
 * Each entry's list of mutable maps is converted to a list of immutable maps.
 *
 * @return A map with the same keys where each value is a `List<Map<String, String>>` (immutable maps).
 */
internal fun Map<String, List<MutableMap<String, String>>>.toImmutableContextRecords(): Map<String, List<Map<String, String>>> =
    mapValues { (_, records) -> records.map { it.toMap() } }

/**
 * Converts each context record map in the receiver to a mutable map, preserving outer keys and list order.
 *
 * @return A map with the same keys where each value is a list of mutable copies of the original maps.
 */
internal fun Map<String, List<Map<String, String>>>.toMutableContextRecords(): Map<String, List<MutableMap<String, String>>> =
    mapValues { (_, records) -> records.map { it.toMutableMap() } }
