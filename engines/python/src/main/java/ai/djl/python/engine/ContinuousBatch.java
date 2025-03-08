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

import ai.djl.Model;
import ai.djl.metric.Dimension;
import ai.djl.metric.Metrics;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.util.PairList;
import ai.djl.util.RandomUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

class ContinuousBatch {

    private static final Logger logger = LoggerFactory.getLogger(ContinuousBatch.class);
    private static final Logger MODEL_METRIC = LoggerFactory.getLogger("model_metric");

    private Dimension dimension;
    private Metrics metrics;
    private Map<String, Request> activeRequests;
    private PyProcess process;

    ContinuousBatch(PyProcess process, Model model) {
        this.dimension = new Dimension("Model", model.getProperty("metric_dimension", "model"));
        this.activeRequests = new ConcurrentHashMap<>();
        this.process = process;
        if (Boolean.parseBoolean(model.getProperty("log_request_metric"))) {
            int metricsAggregation = model.intProperty("metrics_aggregation", 1000);
            metrics = new Metrics();
            metrics.setLimit(metricsAggregation);
            metrics.setOnLimit(
                    (m, s) -> {
                        MODEL_METRIC.info("{}", m.percentile(s, 50));
                        MODEL_METRIC.info("{}", m.percentile(s, 90));
                    });
        }
    }

    Output addInput(Input input) {
        String requestId = input.getProperty("requestId", null);
        assert requestId != null;
        String seed = String.valueOf(RandomUtils.nextInt());
        Request request = new Request(input, seed, metrics, dimension);
        activeRequests.put(requestId, request);
        logger.info("Adding continuous batch request {}", requestId);
        process.sendRequest(input);
        return request.output;
    }

    void addOutput(Output output) {
        PairList<String, BytesSupplier> content = output.getContent();
        assert content.size() == 1;
        // TODO: optimize for conditional killing
        Map<String, String> prop = output.getProperties();
        logger.info("Received output from python with properties {}", prop);
        byte[] responseContent = content.get(0).getValue().getAsBytes();
        String requestId = prop.get("requestId");
        assert requestId != null;
        Request request = activeRequests.get(requestId);
        request.addResponse(responseContent, prop);
        if (request.last) {
            logger.info("Request [{}] completed", requestId);
            activeRequests.remove(requestId);
        }
    }
}
