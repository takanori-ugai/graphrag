package com.microsoft.graphrag.evaluation

private const val PUNCTUATION_CHARS = "!\"#$%&'()*+,-./:;<>?@[\\]^_`{|}~"
private val ARTICLES_REGEX = Regex("\\b(a|an|the)\\b")
private val WHITESPACE_REGEX = Regex("\\s+")

/**
 * Normalizes an answer by lowercasing, removing punctuation/articles, and collapsing whitespace.
 */
fun normalizeAnswer(answer: String): String {
    fun removeArticles(text: String): String = ARTICLES_REGEX.replace(text, " ")

    fun whiteSpaceFix(text: String): String =
        text
            .trim()
            .split(WHITESPACE_REGEX)
            .filter { it.isNotEmpty() }
            .joinToString(" ")

    fun removePunc(text: String): String = text.filter { ch -> ch !in PUNCTUATION_CHARS }

    fun lower(text: String): String = text.lowercase()

    return whiteSpaceFix(removeArticles(removePunc(lower(answer))))
}
