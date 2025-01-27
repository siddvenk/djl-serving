/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.python.engine;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.translate.TranslateException;

/** Interface describing rolling batch inference. */
public interface RollingBatch {

    /**
     * Submits a new batch for the python worker.
     *
     * @param input the batch of requests
     * @param timeout how long to wait for an empy spot in the queue
     * @return the output for the requests
     * @throws TranslateException if python worker capacity is full
     */
    Output addInput(Input input, int timeout) throws TranslateException;

    /** Shuts down the rolling batch worker. */
    void shutdown();
}
