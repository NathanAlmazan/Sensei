package org.nathanael.sensei.store;

import org.nathanael.sensei.Sensei;

public interface ModelStorage {

    Sensei loadModel(String location, Float learningRate) throws Exception;

    void saveModel(Sensei model, String location) throws Exception;

}
