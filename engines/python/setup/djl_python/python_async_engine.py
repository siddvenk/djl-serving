#!/usr/bin/env python
#
# Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import asyncio
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Thread
from queue import Queue
from asyncio.queues import Queue as AsyncQueue
from typing import AsyncGenerator

from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.python_sync_engine import PythonSyncEngine


class PythonAsyncEngine(PythonSyncEngine):
    """
    Backend engine to run python code in decoupled/async mode.
    Requests are forwarded from the model server and submitted to the handler.
    The handler returns responses as they become available, and sends them to the frontend.
    This assumes the frontend maintains tracking of requests and can attribute a response to a given request.
    """

    def __init__(self, args, service):
        super().__init__(args, service)
        self.output_queue = AsyncQueue()
        self.exception_queue = Queue()
        self.loop = None
        # Todo: for async mode we should maybe consider

    def receive_requests(self):
        logging.info("starting receive requests thread")
        while True:
            logging.info("receive loop start")
            inputs, function_name = self._prepare_inputs()

            logging.info("submitting inference task to handler")
            asyncio.run_coroutine_threadsafe(
                self.invoke_handler_async(function_name, inputs), self.loop)

    async def invoke_handler_async(self, function_name: str, inputs: Input):
        try:
            outputs = await self.service.invoke_handler_async(
                function_name, inputs)
            if outputs is None:
                outputs = Output(code=204, message="No content")
            elif not isinstance(outputs, Output):
                outputs = Output().error(
                    f"Invalid output type from handler: {type(outputs)}. Expected type Output.")
        except Exception as e:
            logging.exception("Failed invoke service.invoke_handler_async()")
            if (type(e).__name__ == "OutOfMemoryError"
                    or type(e).__name__ == "MemoryError"
                    or "No available memory for the cache blocks" in str(e)
                    or "CUDA error: out of memory" in str(e)):
                outputs = Output(code=507, message=str(e))
            else:
                outputs = Output().error(str(e), message="service.invoke_handler_async() failure")
        logging.info(f"putting result of inference to output queue")
        await self.output_queue.put(outputs)

    def send_responses(self):
        logging.info("starting send responses thread")
        while True:
            future = asyncio.run_coroutine_threadsafe(self.output_queue.get(),
                                                      self.loop)
            logging.info("waiting for new inference response")
            output: Output = future.result()
            if isinstance(output, AsyncGenerator):
                logging.info("sending streaming response back")
                asyncio.run_coroutine_threadsafe(
                    self.send_stream_response(output), self.loop)
            else:
                logging.info("sending non-streaming response back")
                output.send(self.cl_socket)

    async def send_stream_response(self,
                                   async_generator: AsyncGenerator[Output,
                                                                   None]):
        async for chunk in async_generator:
            chunk.send(self.cl_socket)

    def run_server(self):

        async def main():
            self.loop = asyncio.get_running_loop()
            self._create_cl_socket()

            def catch_all(func):
                try:
                    func()
                except Exception as e:
                    logging.error(f"{func} failed. Details {e}")
                    self.exception_queue.put(str(traceback.format_exc()))

            threads = [
                Thread(target=partial(catch_all, self.receive_requests)),
                Thread(target=partial(catch_all, self.send_responses)),
            ]

            for t in threads:
                t.start()

            def check_threads():
                while True:
                    if not all(t.is_alive() for t in threads):
                        return
                    time.sleep(1)

            with ThreadPoolExecutor(1) as executor:
                await asyncio.get_event_loop().run_in_executor(
                    executor, check_threads)

        asyncio.get_event_loop().run_until_complete(main())
        logging.info("djl async engine terminated")
        if not self.exception_queue.empty():
            logging.error(f"djl async engine terminated with error {self.exception_queue.get()}")
        return None
