package com.microsoft.graphrag.index

/**
 * Returns the list stored at [key] filtered to elements of type [T], or an empty list if absent or mismatched.
 */
inline fun <reified T> Map<String, Any?>.getList(key: String): List<T> = (this[key] as? List<*>)?.filterIsInstance<T>() ?: emptyList()

/**
 * Returns the value for [key] cast to [T], or `null` if the key is absent or the value is not of type [T].
 *
 * Warning: this is not safe for generic types due to type erasure (for example, `List<String>`).
 * Use [getList] for list retrieval to avoid potential runtime `ClassCastException`s.
 */
@Suppress("UNCHECKED_CAST")
inline fun <reified T> Map<String, Any?>.getTypedValue(key: String): T? = this[key] as? T
