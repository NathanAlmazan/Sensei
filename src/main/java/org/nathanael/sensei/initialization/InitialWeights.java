package org.nathanael.sensei.initialization;


import org.nathanael.sensei.activations.*;

public abstract class InitialWeights {
    public static WeightInitialization getDefaultInitialization(Activation activation) {
        if (activation instanceof LeakyReLu || activation instanceof ReLu) return new HeInitial();
        else if (activation instanceof Sigmoid) return new NormalizedXavier();
        else if (activation instanceof Softmax) return new Xavier();
        else return new NormalRandom();
    }
}
