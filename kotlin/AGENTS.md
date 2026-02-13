# Agent Notes (kotlin)

## Project overview
- Kotlin JVM application using Ktor, Koin, LangChain4j, and JavaFX.
- Main entrypoint defaults to `com.microsoft.graphrag.cli.GraphRagCliKt`.
- Gradle Kotlin DSL build with ktlint and detekt.
- Kotlin codebase converted from a Python program. LLM output format tends to be JSON.

## Common commands
- Build: `./gradlew build`
- Test: `./gradlew test`
- Lint: `./gradlew ktlintCheck`
- Detekt: `./gradlew detekt`
- Run (default main): `./gradlew run`
- Run (explicit main): `./gradlew execute -PmainClass=fully.qualified.ClassKt`

## Code style
- Prefer idiomatic Kotlin and keep formatting compatible with ktlint.
- Keep public APIs documented and minimize changes to behavior unless required.
- Avoid editing generated output under `build/`.

## Repo structure
- `src/` application code.
- `config/` static config (detekt rules, etc.).
- `settings.sample.yaml` example configuration.

## Testing notes
- Tests run under JUnit via `kotlin-test-junit` and may log to stdout/stderr.
