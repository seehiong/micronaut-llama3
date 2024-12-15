package example.micronaut.model;

import java.util.Map;

import example.micronaut.datatype.GGUFTensorInfo;

public record PartialModel(String modelFileName, Llama model, long tensorDataOffset,
        Map<String, GGUFTensorInfo> tensorInfos) {

}
