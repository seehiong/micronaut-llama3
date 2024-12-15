package example.micronaut.datatype;

public record GGUFTensorInfo(String name, int[] dimensions, GGMLType ggmlType, long offset) {
}
