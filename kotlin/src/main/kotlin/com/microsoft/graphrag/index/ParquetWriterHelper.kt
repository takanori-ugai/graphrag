package com.microsoft.graphrag.index

import blue.strategic.parquet.Dehydrator
import blue.strategic.parquet.ParquetWriter
import blue.strategic.parquet.ValueWriter
import org.apache.parquet.schema.LogicalTypeAnnotation
import org.apache.parquet.schema.MessageType
import org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName
import org.apache.parquet.schema.Types
import java.io.File
import java.nio.file.Files
import java.nio.file.Path

/**
 * Writes index artifacts to Parquet files.
 */
@Suppress("TooManyFunctions")
class ParquetWriterHelper {
    /**
     * Writes entity records to a Parquet file.
     *
     * @param path Output file path.
     * @param entities Entities to write.
     */
    fun writeEntities(
        path: Path,
        entities: List<Entity>,
    ) {
        val schema =
            message("Entity") {
                addField(requiredString("id"))
                addField(requiredString("name"))
                addField(requiredString("type"))
                addField(requiredString("sourceChunkId"))
            }
        write(path, schema, entities) { e, w ->
            w.write("id", e.id)
            w.write("name", e.name)
            w.write("type", e.type)
            w.write("sourceChunkId", e.sourceChunkId)
        }
    }

    /**
     * Writes relationship records to a Parquet file.
     *
     * @param path Output file path.
     * @param relationships Relationships to write.
     */
    fun writeRelationships(
        path: Path,
        relationships: List<Relationship>,
    ) {
        val schema =
            message("Relationship") {
                addField(requiredString("sourceId"))
                addField(requiredString("targetId"))
                addField(requiredString("type"))
                addField(optionalString("description"))
                addField(requiredString("sourceChunkId"))
            }
        write(path, schema, relationships) { r, w ->
            w.write("sourceId", r.sourceId)
            w.write("targetId", r.targetId)
            w.write("type", r.type)
            w.write("description", r.description ?: "")
            w.write("sourceChunkId", r.sourceChunkId)
        }
    }

    /**
     * Writes text embeddings to a Parquet file.
     *
     * @param path Output file path.
     * @param embeddings Text embeddings to write.
     */
    fun writeTextEmbeddings(
        path: Path,
        embeddings: List<TextEmbedding>,
    ) {
        val schema =
            message("TextEmbedding") {
                addField(requiredString("chunkId"))
                addField(repeatedDouble("vector"))
            }
        write(path, schema, embeddings) { e, w ->
            w.write("chunkId", e.chunkId)
            e.vector.forEach { v -> w.write("vector", v) }
        }
    }

    /**
     * Writes entity embeddings to a Parquet file.
     *
     * @param path Output file path.
     * @param embeddings Entity embeddings to write.
     */
    fun writeEntityEmbeddings(
        path: Path,
        embeddings: List<EntityEmbedding>,
    ) {
        val schema =
            message("EntityEmbedding") {
                addField(requiredString("entityId"))
                addField(repeatedDouble("vector"))
            }
        write(path, schema, embeddings) { e, w ->
            w.write("entityId", e.entityId)
            e.vector.forEach { v -> w.write("vector", v) }
        }
    }

    /**
     * Writes community assignments to a Parquet file.
     *
     * @param path Output file path.
     * @param assignments Community assignments to write.
     */
    fun writeCommunityAssignments(
        path: Path,
        assignments: List<CommunityAssignment>,
    ) {
        val schema =
            message("CommunityAssignment") {
                addField(requiredString("entityId"))
                addField(requiredInt("communityId"))
            }
        write(path, schema, assignments) { c, w ->
            w.write("entityId", c.entityId)
            w.write("communityId", c.communityId)
        }
    }

    /**
     * Writes claims to a Parquet file.
     *
     * @param path Output file path.
     * @param claims Claims to write.
     */
    fun writeClaims(
        path: Path,
        claims: List<Claim>,
    ) {
        val schema =
            message("Claim") {
                addField(requiredString("subject"))
                addField(requiredString("object"))
                addField(requiredString("claimType"))
                addField(requiredString("status"))
                addField(requiredString("startDate"))
                addField(requiredString("endDate"))
                addField(requiredString("description"))
                addField(requiredString("sourceText"))
            }
        write(path, schema, claims) { c, w ->
            w.write("subject", c.subject)
            w.write("object", c.`object`)
            w.write("claimType", c.claimType)
            w.write("status", c.status)
            w.write("startDate", c.startDate)
            w.write("endDate", c.endDate)
            w.write("description", c.description)
            w.write("sourceText", c.sourceText)
        }
    }

    /**
     * Writes text units to a Parquet file.
     *
     * @param path Output file path.
     * @param textUnits Text units to write.
     */
    fun writeTextUnits(
        path: Path,
        textUnits: List<TextUnit>,
    ) {
        val schema =
            message("TextUnit") {
                addField(requiredString("id"))
                addField(requiredString("chunkId"))
                addField(requiredString("text"))
                addField(requiredString("sourcePath"))
            }
        write(path, schema, textUnits) { t, w ->
            w.write("id", t.id)
            w.write("chunkId", t.chunkId)
            w.write("text", t.text)
            w.write("sourcePath", t.sourcePath)
        }
    }

    /**
     * Writes entity summaries to a Parquet file.
     *
     * @param path Output file path.
     * @param summaries Entity summaries to write.
     */
    fun writeEntitySummaries(
        path: Path,
        summaries: List<EntitySummary>,
    ) {
        val schema =
            message("EntitySummary") {
                addField(requiredString("entityId"))
                addField(requiredString("summary"))
            }
        write(path, schema, summaries) { s, w ->
            w.write("entityId", s.entityId)
            w.write("summary", s.summary)
        }
    }

    private fun requiredString(name: String) = Types.required(PrimitiveTypeName.BINARY).`as`(LogicalTypeAnnotation.stringType()).named(name)

    private fun optionalString(name: String) = Types.optional(PrimitiveTypeName.BINARY).`as`(LogicalTypeAnnotation.stringType()).named(name)

    private fun repeatedDouble(name: String) = Types.repeated(PrimitiveTypeName.DOUBLE).named(name)

    private fun requiredInt(name: String) = Types.required(PrimitiveTypeName.INT32).named(name)

    private fun message(
        name: String,
        build: Types.MessageTypeBuilder.() -> Unit,
    ): MessageType {
        val builder = Types.buildMessage()
        build(builder)
        return builder.named(name)
    }

    private fun <T> write(
        path: Path,
        schema: MessageType,
        records: List<T>,
        dehydrator: (T, ValueWriter) -> Unit,
    ) {
        ensureParent(path)
        val pw =
            ParquetWriter.writeFile(
                schema,
                File(path.toString()),
                Dehydrator<T> { record, valueWriter ->
                    dehydrator(record, valueWriter)
                },
            )
        pw.use { writer -> records.forEach { writer.write(it) } }
    }

    private fun ensureParent(path: Path) {
        val parent = path.parent ?: return
        if (!Files.exists(parent)) {
            Files.createDirectories(parent)
        }
    }
}
