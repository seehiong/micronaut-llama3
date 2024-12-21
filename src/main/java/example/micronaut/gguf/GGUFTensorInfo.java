package example.micronaut.gguf;

public record GGUFTensorInfo(String name, int[] dimensions, GGMLType ggmlType, long offset) {
}
