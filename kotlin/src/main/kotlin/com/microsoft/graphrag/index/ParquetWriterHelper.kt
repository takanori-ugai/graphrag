package com.microsoft.graphrag.index

import org.apache.avro.Schema
import org.apache.avro.SchemaBuilder
import org.apache.avro.generic.GenericData
import org.apache.parquet.avro.AvroParquetWriter
import org.apache.parquet.hadoop.metadata.CompressionCodecName
import org.apache.parquet.io.OutputFile
import org.apache.parquet.io.PositionOutputStream
import java.nio.file.Files
import java.nio.file.Path

class ParquetWriterHelper {
    private val entitySchema: Schema =
        SchemaBuilder
            .record("Entity")
            .fields()
            .name("id")
            .type()
            .stringType()
            .noDefault()
            .name("name")
            .type()
            .stringType()
            .noDefault()
            .name("type")
            .type()
            .stringType()
            .noDefault()
            .name("sourceChunkId")
            .type()
            .stringType()
            .noDefault()
            .endRecord()

    private val relationshipSchema: Schema =
        SchemaBuilder
            .record("Relationship")
            .fields()
            .name("sourceId")
            .type()
            .stringType()
            .noDefault()
            .name("targetId")
            .type()
            .stringType()
            .noDefault()
            .name("type")
            .type()
            .stringType()
            .noDefault()
            .name("description")
            .type()
            .nullable()
            .stringType()
            .noDefault()
            .name("sourceChunkId")
            .type()
            .stringType()
            .noDefault()
            .endRecord()

    fun writeEntities(
        path: Path,
        entities: List<Entity>,
    ) {
        ensureParent(path)
        AvroParquetWriter
            .builder<GenericData.Record>(LocalOutputFile(path))
            .withSchema(entitySchema)
            .withCompressionCodec(CompressionCodecName.SNAPPY)
            .build()
            .use { writer ->
                entities.forEach { entity ->
                    writer.write(
                        GenericData.Record(entitySchema).apply {
                            put("id", entity.id)
                            put("name", entity.name)
                            put("type", entity.type)
                            put("sourceChunkId", entity.sourceChunkId)
                        },
                    )
                }
            }
    }

    fun writeRelationships(
        path: Path,
        relationships: List<Relationship>,
    ) {
        ensureParent(path)
        AvroParquetWriter
            .builder<GenericData.Record>(LocalOutputFile(path))
            .withSchema(relationshipSchema)
            .withCompressionCodec(CompressionCodecName.SNAPPY)
            .build()
            .use { writer ->
                relationships.forEach { rel ->
                    writer.write(
                        GenericData.Record(relationshipSchema).apply {
                            put("sourceId", rel.sourceId)
                            put("targetId", rel.targetId)
                            put("type", rel.type)
                            put("description", rel.description)
                            put("sourceChunkId", rel.sourceChunkId)
                        },
                    )
                }
            }
    }

    private fun ensureParent(path: Path) {
        val parent = path.parent
        if (parent != null && !Files.exists(parent)) {
            Files.createDirectories(parent)
        }
    }
}

private class LocalOutputFile(
    private val path: Path,
) : OutputFile {
    override fun create(blockSizeHint: Long): PositionOutputStream = newStream()

    override fun createOrOverwrite(blockSizeHint: Long): PositionOutputStream = newStream()

    override fun supportsBlockSize(): Boolean = false

    override fun defaultBlockSize(): Long = 0

    private fun newStream(): PositionOutputStream =
        object : PositionOutputStream() {
            private val channel = Files.newOutputStream(path)
            private var position: Long = 0

            override fun getPos(): Long = position

            override fun write(b: Int) {
                channel.write(byteArrayOf(b.toByte()))
                position += 1
            }

            override fun write(
                b: ByteArray,
                off: Int,
                len: Int,
            ) {
                channel.write(b, off, len)
                position += len.toLong()
            }

            override fun close() {
                channel.close()
            }
        }
}
