from typing import List, Tuple, Union
import tensorflow as tf
from morphocut import Node, Output, RawOrVariable, ReturnOutputs, closing_if_closable
from morphocut.batch import Batch
from morphocut.utils import buffered_generator


@ReturnOutputs
@Output("output")
class TensorFlow(Node):
    """
    Apply a TensorFlow module to the input.

    Args:
        module (tf.Module): TensorFlow module.
        input (input, tf.Tensor): Input.
        device (str or tf.device, optional): Device.
        n_parallel (int, optional): Run multiple computations in parallel.
            0 means synchronous computations.
        is_batch (bool, optional): Assume that input is a batch.
        output_key (optional): If the module has multiple outputs, output_key selects one of them.
    """

    def __init__(
        self,
        module: tf.Module,
        input: RawOrVariable,
        device=None,
        n_parallel=0,
        is_batch=True,
        output_key=None,
    ):
        super().__init__()

        print("TensorFlow.device: ", device)

        if device is not None:
            with tf.device(device):
                self.model = module
        else:
            self.model = module

        self.input = input
        self.n_parallel = n_parallel
        self.is_batch = is_batch
        self.output_key = output_key

    def transform_stream(self, stream):
        @buffered_generator(self.n_parallel)
        def output_gen():
            with closing_if_closable(stream):
                for obj in stream:
                    input = self.prepare_input(obj, "input")

                    # Assemble batch
                    if isinstance(input, Batch):
                        input = tf.stack(input)
                    elif not self.is_batch:
                        input = tf.expand_dims(input, 0)

                    if self.model.variables:
                        # Enable evaluation mode
                        self.model.eval()

                        output = self.model(input, training=False)
                    else:
                        output = self.model(input)

                    if self.output_key is not None:
                        output = output[self.output_key]

                    yield obj, output

        for obj, output in output_gen():
            output = output.numpy()

            if not self.is_batch:
                output = output[0]

            yield self.prepare_output(obj, output)
