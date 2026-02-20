package com.microsoft.graphrag.index

/**
 * Retrieve elements of type T from a list stored under the given map key.
 *
 * @param key The map key whose value is expected to be a list.
 * @return A `List<T>` containing elements of type T from the value at `key`; an empty list
 * if the key is absent, the value is not a list, or no elements match.
 */
inline fun <reified T> Map<String, Any?>.getList(key: String): List<T> = (this[key] as? List<*>)?.filterIsInstance<T>() ?: emptyList()

/**
 * Retrieve the value for the given key and return it as `T` when the value is present and of that type.
 *
 * Due to type erasure this cannot reliably check generic parameterized types
 * (for example `List<String>`); use `getList` for lists.
 *
 * @param key The map key whose associated value should be returned as `T`.
 * @return The value cast to `T` if present and of that type, `null` otherwise.
 */
@Suppress("UNCHECKED_CAST")
inline fun <reified T> Map<String, Any?>.getTypedValue(key: String): T? = this[key] as? T
