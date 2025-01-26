package ai.djl.python.engine;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.translate.TranslateException;

public interface RollingBatch {

    Output addInput(Input input, int timeout) throws TranslateException;
    void shutdown();
}
