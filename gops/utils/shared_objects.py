#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Some shared objects classes for distributed RL framework
#  Update: 2023-09-12, Xiangyu Zhu: create codes

import torch
import pprint
import numpy as np
import multiprocessing as mp
from typing import Dict
from collections import OrderedDict


numpy_to_torch_dtype_dict = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128
}
torch_to_numpy_dtype_dict = {v: k for k, v in numpy_to_torch_dtype_dict.items()}

numpy_to_torch_dtype_dict.update({np.dtype(k): v for k, v in numpy_to_torch_dtype_dict.items()})


class SharedArray:
    """A SharedArray class.

    :param tuple shape: The shape of the shared array.
    :param Optional[numpy.dtype] dtype: The data type of the shared array. Defaults to np.float32.
    """
    def __init__(self, shape: tuple, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype

        self.tensor = torch.zeros(shape, dtype=numpy_to_torch_dtype_dict[dtype])
        self.tensor.share_memory_()
        assert self.tensor.is_shared()

    def pin_memory_(self, device):
        """Pins the memory of the tensor to the specified CUDA device.

        :param torch.device device: The CUDA device to pin the memory to.

        Returns: None

        Raises: AssertionError: If the memory pinning fails.
        """
        with torch.cuda.device(device):
            result = torch.cuda.cudart().cudaHostRegister(self.tensor.data_ptr(),
                                                          self.tensor.numel() * self.tensor.element_size(), 0)
            assert result.value == 0, "Failed to pin memory."

        assert self.tensor.is_pinned()

    def get(self):
        return self.tensor.numpy()

    def get_torch(self):
        return self.tensor

    def __repr__(self):
        return "<SharedArray shape={}, dtype={}>".format(self.shape, self.dtype)


class BatchLocal:
    pass


class BatchShared:
    """Initialize a BatchShared object.

    :param Dict shape_dtype: A dictionary specifying the shape and dtype of the shared arrays.
        The keys are the names of the arrays, and the values are tuples of shape and dtype.
    :param bool init_ready: Flag indicating whether the BatchShared object should be initialized as ready.
    """
    def __init__(self, shape_dtype: Dict, init_ready: bool) -> None:
        self.data = {k: SharedArray(*v) for k, v in shape_dtype.items()}
        self.ready = mp.BoundedSemaphore(1)
        if not init_ready:
            self.ready.acquire()

    def get(self) -> BatchLocal:
        """Get the data stored in the BatchShared object.

        :return: A BatchLocal object containing the data.
        """
        obj = BatchLocal()
        obj.__dict__.update({k: v.get() for k, v in self.data.items()})
        return obj

    def set_ready(self) -> None:
        """Set the BatchShared object as ready."""
        self.ready.release()

    def wait_ready(self) -> None:
        """Wait until the BatchShared object is ready."""
        self.ready.acquire()

    def __repr__(self):
        return "<BatchShared: {}>".format(pprint.pformat(self.data))


class BatchCuda:
    """A class that represents a batch of data on CUDA device.

    Args:
        batch_shared (BatchShared): The batch shared object that contains the data.
        device (torch.device): The CUDA device where the data is stored.

    Attributes:
        batch_shared (BatchShared): The batch shared object that contains the data.
        device (torch.device): The CUDA device where the data is stored.
        src (dict): A dictionary of tensors that are pinned on the CPU and copied to the CUDA device.
        data (dict): A dictionary of tensors that are created on the CUDA device and pinned there.
    """
    def __init__(self, batch_shared: BatchShared, device: torch.device) -> None:
        self.batch_shared = batch_shared
        self.device = device

        # Source
        if device.type == "cuda":
            # Pin memory if on CUDA
            [v.pin_memory_(device) for v in self.batch_shared.data.values()]

        self.src = {k: v.get_torch() for k, v in self.batch_shared.data.items()}
        # CUDA buffer
        self.data = {k: torch.zeros(v.shape, dtype=numpy_to_torch_dtype_dict[v.dtype], device=device)
                     for k, v in self.batch_shared.data.items()}

    def copy_from(self) -> None:
        """Copies the data from CPU to CUDA.

        This method will copy all the tensors from the src dictionary to the corresponding tensors
        in the data dictionary. It will also pin them on the CUDA device. This method is asynchronous
        and does not block the caller thread.
        """
        # (Async) CPU pinned --> CUDA
        for k, v in self.data.items():
            v.copy_(self.src[k], non_blocking=True)
        # Wait all copy ready
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def wait_ready(self) -> None:
        """Waits until all tensors are ready on the CUDA device.

        This method will block until all tensors in the data dictionary are copied from CPU to CUDA and pinned there.
        It will also wait for any pending operations on the CUDA device to complete.
        """
        self.batch_shared.wait_ready()

    def get_batch(self) -> Dict:
        batch = dict()
        batch["obs"] = self.data["obs"][:, -2]
        batch["obs2"] = self.data["obs"][:, -1]
        batch["act"] = self.data["act"][: -2]
        batch["rew"] = self.data["rew"][:, -2]
        batch["done"] = self.data["done"][:, -2]
        return batch


class SharedStateDict:
    """A shared state_dict that has one publisher and many subscribers.

    Args:
        state_dict (dict): The source shared state_dict that contains the data.
    """
    def __init__(self, state_dict: OrderedDict) -> None:
        self.shared_cpu = OrderedDict((k, SharedArray(v.shape, torch_to_numpy_dtype_dict[v.dtype]))
                                       for k, v in state_dict.items())

        self.type = None
        self.device = None
        self.shared_cpu_tensor = None

    def initialize(self, type: str, device: torch.device) -> None:
        """Initializes the shared state_dict as either publisher or subscriber.

        Args:
            type (str): The type of the shared state_dict. Must be either "publisher" or "subscriber".
            device (torch.device): The device where the shared state_dict will be stored.

        Raises:
            AssertionError: If the type is not "publisher" or "subscriber".
        """
        self.type = type
        self.device = device
        self.shared_cpu_tensor = OrderedDict((k, v.get_torch()) for k, v in self.shared_cpu.items())

        if device.type == "cuda":
            if type == "publisher":
                # Pin memory
                [v.pin_memory_(device) for v in self.shared_cpu.values()]
            elif type == "subscriber":
                # Create pinned memory buffer
                with torch.cuda.device(device):
                    self.local_cpu_buffer = OrderedDict(
                        (k, torch.zeros_like(v, pin_memory=True)) for k, v in self.shared_cpu_tensor.items())

    def copy_state_dict(self, src: Dict, dst: Dict) -> None:
        """Copies the data from one shared state_dict to another.

        Args:
            src (dict): The source shared state_dict that contains the data.
            dst (dict): The destination shared state_dict that will receive the data.
        """
        with torch.no_grad():
            for k, v in dst.items():
                v.copy_(src[k], non_blocking=True)

    def publish(self, state_dict: Dict) -> None:
        """Publishes the data from one shared state_dict to another.

        Args:
            state_dict (dict): The source shared state_dict that contains the data.
        """
        assert self.type == "publisher", "SharedStateDict: Must initialize as publisher"

        self.copy_state_dict(state_dict, self.shared_cpu_tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def receive(self, state_dict: Dict) -> None:
        """Receives the data from another shared state_dict to this one.

        Args:
            state_dict (dict): The source shared state_dict that contains the data.
        """
        assert self.type == "subscriber", "SharedStateDict: Must initialize as subscriber"

        if self.device.type == "cuda":
            # Shared --> pinned buffer --> device
            self.copy_state_dict(self.shared_cpu_tensor, self.local_cpu_buffer)
            self.copy_state_dict(self.local_cpu_buffer, state_dict)
        else:
            # Direct copy
            self.copy_state_dict(self.shared_cpu_tensor, state_dict)

    def __repr__(self):
        return "<SharedStateDict {}>".format(list(self.shared_cpu.keys()))