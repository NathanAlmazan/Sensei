package org.nathanael.sensei.store;

import org.nathanael.sensei.activations.*;
import org.nathanael.sensei.initialization.*;
import org.nathanael.sensei.loss.BinaryCrossEntropy;
import org.nathanael.sensei.loss.LossFunction;
import org.nathanael.sensei.loss.MeanSquareError;

import java.util.HashMap;
import java.util.Map;

public abstract class StringToObject {
    public static HashMap<String, Activation> ACTIVATION = new HashMap<>(Map.of(
            "LeakyReLu", new LeakyReLu(),
            "ReLu", new ReLu(),
            "Sigmoid", new Sigmoid(),
            "Softmax", new Softmax(),
            "Tanh", new Tanh()
    ));

    public static HashMap<String, LossFunction> LOSS_FUNCTION = new HashMap<>(Map.of(
            "MeanSquareError", new MeanSquareError(),
            "BinaryCrossEntropy", new BinaryCrossEntropy()
    ));

    public static HashMap<String, WeightInitialization> INITIALIZATION = new HashMap<>(Map.of(
            "HeInitial", new HeInitial(),
            "Xavier", new Xavier(),
            "NormalizedXavier", new NormalizedXavier(),
            "NormalRandom", new NormalRandom()
    ));
}
