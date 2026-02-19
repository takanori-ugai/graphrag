package com.microsoft.graphrag.index

@Suppress("UNCHECKED_CAST")
inline fun <reified T> Map<String, Any?>.getList(key: String): List<T> = (this[key] as? List<*>)?.filterIsInstance<T>() ?: emptyList()

@Suppress("UNCHECKED_CAST")
inline fun <reified T> Map<String, Any?>.getValue(key: String): T? = this[key] as? T
