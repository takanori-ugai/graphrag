package com.microsoft.graphrag.evaluation

import kotlin.test.Test
import kotlin.test.assertEquals

class EvalUtilsTest {
    @Test
    fun `normalizeAnswer should lowercase text`() {
        val input = "Hello World"
        val expected = "hello world"
        assertEquals(expected, normalizeAnswer(input))
    }

    @Test
    fun `normalizeAnswer should remove punctuation`() {
        val input = "Hello, World!"
        val expected = "hello world"
        assertEquals(expected, normalizeAnswer(input))
    }

    @Test
    fun `normalizeAnswer should remove articles`() {
        val input = "The quick brown fox jumps over a lazy dog"
        val expected = "quick brown fox jumps over lazy dog"
        assertEquals(expected, normalizeAnswer(input))
    }

    @Test
    fun `normalizeAnswer should collapse multiple spaces`() {
        val input = "Hello    World   with    spaces"
        val expected = "hello world with spaces"
        assertEquals(expected, normalizeAnswer(input))
    }

    @Test
    fun `normalizeAnswer should handle empty string`() {
        val input = ""
        val expected = ""
        assertEquals(expected, normalizeAnswer(input))
    }

    @Test
    fun `normalizeAnswer should handle string with only punctuation`() {
        val input = "!@#$%^&*()"
        val expected = ""
        assertEquals(expected, normalizeAnswer(input))
    }

    @Test
    fun `normalizeAnswer should handle string with only articles`() {
        val input = "a an the"
        val expected = ""
        assertEquals(expected, normalizeAnswer(input))
    }

    @Test
    fun `normalizeAnswer should handle string with only whitespace`() {
        val input = "   \t\n  "
        val expected = ""
        assertEquals(expected, normalizeAnswer(input))
    }

    @Test
    fun `normalizeAnswer should process complex sentence`() {
        val input = "The answer is: 42! It's a number."
        val expected = "answer is 42 its number"
        assertEquals(expected, normalizeAnswer(input))
    }

    @Test
    fun `normalizeAnswer should handle text with mixed punctuation and articles`() {
        val input = "\"A\" quick, brown 'fox' - jumps!"
        val expected = "quick brown fox jumps"
        assertEquals(expected, normalizeAnswer(input))
    }

    @Test
    fun `normalizeAnswer should preserve numbers`() {
        val input = "The answer is 123"
        val expected = "answer is 123"
        assertEquals(expected, normalizeAnswer(input))
    }

    @Test
    fun `normalizeAnswer should handle unicode characters`() {
        val input = "Café résumé naïve"
        val expected = "café résumé naïve"
        assertEquals(expected, normalizeAnswer(input))
    }

    @Test
    fun `normalizeAnswer should handle text starting with article`() {
        val input = "An apple a day"
        val expected = "apple day"
        assertEquals(expected, normalizeAnswer(input))
    }

    @Test
    fun `normalizeAnswer should handle text ending with article`() {
        val input = "Give me a"
        val expected = "give me"
        assertEquals(expected, normalizeAnswer(input))
    }

    @Test
    fun `normalizeAnswer should handle consecutive punctuation`() {
        val input = "Hello!!!...World???"
        val expected = "helloworld"
        assertEquals(expected, normalizeAnswer(input))
    }

    @Test
    fun `normalizeAnswer should trim leading and trailing whitespace`() {
        val input = "  Hello World  "
        val expected = "hello world"
        assertEquals(expected, normalizeAnswer(input))
    }

    @Test
    fun `normalizeAnswer should handle newlines and tabs`() {
        val input = "Hello\n\tWorld"
        val expected = "hello world"
        assertEquals(expected, normalizeAnswer(input))
    }

    @Test
    fun `normalizeAnswer should be idempotent`() {
        val input = "The quick, brown fox!"
        val normalized = normalizeAnswer(input)
        val doubleNormalized = normalizeAnswer(normalized)
        assertEquals(normalized, doubleNormalized)
    }
}
