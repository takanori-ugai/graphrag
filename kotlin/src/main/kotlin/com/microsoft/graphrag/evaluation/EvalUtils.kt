package com.microsoft.graphrag.evaluation

private const val PUNCTUATION_CHARS = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
private val ARTICLES_REGEX = Regex("\\b(a|an|the)\\b")
private val WHITESPACE_REGEX = Regex("\\s+")

/**
 * Normalizes an answer by lowercasing, removing punctuation/articles, and collapsing whitespace.
 */
fun normalizeAnswer(answer: String): String {
    /**
 * Removes standalone English articles ("a", "an", "the") from the provided text by replacing each with a single space.
 *
 * @param text The input string to process.
 * @return The input string with each standalone article replaced by a single space.
 */
fun removeArticles(text: String): String = ARTICLES_REGEX.replace(text, " ")

    /**
 * Trim the input string and collapse consecutive whitespace characters into single spaces.
 *
 * @param text The string to normalize.
 * @return The trimmed string with all runs of whitespace replaced by a single space.
 */
fun whiteSpaceFix(text: String): String = WHITESPACE_REGEX.replace(text.trim(), " ")

    /**
 * Removes punctuation characters from the input string.
 *
 * @param text The string to strip of punctuation.
 * @return The input string with all punctuation characters removed.
 */
fun removePunc(text: String): String = text.filter { ch -> ch !in PUNCTUATION_CHARS }

    /**
 * Converts a string to lowercase.
 *
 * @param text The input string to convert.
 * @return The input string with all characters converted to lowercase.
 */
fun lower(text: String): String = text.lowercase()

    return whiteSpaceFix(removeArticles(removePunc(lower(answer))))
}