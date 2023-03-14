package org.nathanael.sensei.dataset;

public interface DatasetLoader {
    Dataset loadDataset(String location) throws Exception;
}
