import torch.nn.functional as F
from bitsandbytes.functional import *
import bitsandbytes.functional as bbf
import bitsandbytes.autograd._functions as bbaf


IS_SWAPPED = False

dequantize_blockwise_old = bbf.dequantize_blockwise
dequantize_4bit_old = bbf.dequantize_4bit
MatMul4BitOld = bbaf.MatMul4Bit


def apply_changes():
    global IS_SWAPPED
    if IS_SWAPPED:
        return
    bbf.dequantize_blockwise = dequantize_blockwise_new
    bbf.dequantize_4bit = dequantize_4bit_new
    bbaf.MatMul4Bit = MatMul4BitNew

    IS_SWAPPED = True


def undo_changes():
    global IS_SWAPPED
    if not IS_SWAPPED:
        return
    bbf.dequantize_blockwise = dequantize_blockwise_old
    bbf.dequantize_4bit = dequantize_4bit_old
    bbaf.MatMul4Bit = MatMul4BitOld
    IS_SWAPPED = False


def unwrap(x):
    if isinstance(x, torch.Tensor):
        if torch._C._functorch.is_gradtrackingtensor(x):
            return torch._C._functorch.get_unwrapped(x)
        else:
            return x
    elif isinstance(x, list):
        return [unwrap(x1) for x1 in x]
    else:
        return x


def dequantize_blockwise_new(
    A: Tensor,
    quant_state: Tuple[Tensor, Tensor] = None,
    absmax: Tensor = None,
    code: Tensor = None,
    out: Tensor = None,
    blocksize: int = 4096,
    nested=False
) -> Tensor:
    """
    Dequantizes blockwise quantized values.

    Dequantizes the tensor A with maximum absolute values absmax in
    blocks of size 4096.

    Parameters
    ----------
    A : torch.Tensor
        The input 8-bit tensor.
    quant_state : tuple(torch.Tensor, torch.Tensor)
        Tuple of code and absmax values.
    absmax : torch.Tensor
        The absmax values.
    code : torch.Tensor
        The quantization map.
    out : torch.Tensor
        Dequantized output tensor (default: float32)


    Returns
    -------
    torch.Tensor:
        Dequantized tensor (default: float32)
    """
    assert quant_state is not None or absmax is not None
    if code is None and quant_state is None:
        if "dynamic" not in name2qmap:
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        code = name2qmap["dynamic"]

    if out is None:
        out = torch.zeros_like(A, dtype=torch.float32)

    if quant_state is None:
       quant_state = (absmax, code, blocksize)
       assert absmax is not None and out is not None
    else:
       absmax, code, blocksize, nested, offset, state2 = quant_state
       if nested:
           absmax = dequantize_blockwise(absmax, state2)
           absmax += offset


    if A.device.type != 'cpu':
        device = pre_call(A.device)
        code = code.to(A.device)
        if blocksize not in [2048, 4096, 1024, 512, 256, 128, 64]:
            raise ValueError(f"The blockwise of {blocksize} is not supported. Supported values: [2048, 4096, 1024, 512, 256, 128, 64]")
        is_on_gpu([A, absmax, out])
        if out.dtype == torch.float32:
            lib.cdequantize_blockwise_fp32(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(blocksize), ct.c_int(A.numel()))
        elif out.dtype == torch.float16:
            lib.cdequantize_blockwise_fp16(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(blocksize), ct.c_int(A.numel()))
        else:
            raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}")
        post_call(A.device)
    else:
        code = code.cpu()
        lib.cdequantize_blockwise_cpu_fp32(get_ptr(quant_state[1]), get_ptr(A), get_ptr(quant_state[0]), get_ptr(out), ct.c_longlong(blocksize), ct.c_longlong(A.numel()))

    return out


def dequantize_4bit_new(A: Tensor,quant_state: Tuple[Tensor, Tensor] = None, absmax: Tensor = None, out: Tensor = None, blocksize: int = 64, quant_type='fp4') -> Tensor:
    """
    Dequantizes FP4 blockwise quantized values.

    Dequantizes the tensor A with maximum absolute values absmax in blocks of size blocksize.

    Parameters
    ----------
    A : torch.Tensor
        The input 8-bit tensor (packed 4-bit values).
    quant_state : tuple(torch.Tensor, torch.Size, torch.dtype)
        Tuple of absmax values, original tensor shape and original dtype.
    absmax : torch.Tensor
        The absmax values.
    out : torch.Tensor
        Dequantized output tensor.
    blocksize : int
        The blocksize used in quantization.
    quant_type : str
        The 4-bit quantization data type {fp4, nf4}


    Returns
    -------
    torch.Tensor:
        Dequantized tensor.
    """
    if blocksize not in [2048, 4096, 1024, 512, 256, 128, 64]:
        raise ValueError(f"The blockwise of {blocksize} is not supported. Supported values: [2048, 4096, 1024, 512, 256, 128, 64]")
    if quant_type not in ['fp4', 'nf4']:
        raise NotImplementedError(f'4-bit quantization data type {quant_type} is not implemented.')

    if quant_state is None:
        assert absmax is not None and out is not None
        shape = out.shape
        dtype = out.dtype
    else:
        absmax, shape, dtype, blocksize, compressed_stats, quant_type = quant_state


    if compressed_stats is not None:
        offset, state2 = compressed_stats
        absmax = dequantize_blockwise(absmax, state2)
        absmax += offset

    if out is None:
        out = torch.empty(shape, dtype=dtype, device=A.device)

    n = out.numel()


    device = pre_call(A.device)
    is_on_gpu([A, absmax, out])
    if out.dtype == torch.float32:
        if quant_type == 'fp4':
            lib.cdequantize_blockwise_fp32_fp4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(blocksize), ct.c_int(n))
        else:
            lib.cdequantize_blockwise_fp32_nf4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(blocksize), ct.c_int(n))
    elif out.dtype == torch.float16:
        if quant_type == 'fp4':
            lib.cdequantize_blockwise_fp16_fp4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(blocksize), ct.c_int(n))
        else:
            lib.cdequantize_blockwise_fp16_nf4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(blocksize), ct.c_int(n))
    else:
        raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}")
    post_call(A.device)

    is_transposed = (True if A.shape[0] == 1 else False)
    if is_transposed: return out.t()
    else: return out


class MatMul4BitNew(torch.autograd.Function):
    # forward is the same, but we added the fallback for pre-turing GPUs
    # backward is mostly the same, but adds one extra clause (see "elif state.CxB is not None")

    @staticmethod
    def forward(ctx, A, B, out=None, bias=None, state=None):
        # default of pytorch behavior if inputs are empty
        ctx.is_empty = False
        if prod(A.shape) == 0:
            ctx.is_empty = True
            ctx.A = A
            ctx.B = B
            ctx.bias = bias
            B_shape = state[1]
            if A.shape[-1] == B_shape[0]:
                return torch.empty(A.shape[:-1] + B_shape[1:], dtype=A.dtype, device=A.device)
            else:
                return torch.empty(A.shape[:-1] + B_shape[:1], dtype=A.dtype, device=A.device)


        # 1. Dequantize
        # 2. MatmulnN
        output = torch.nn.functional.linear(A, F.dequantize_fp4(B, state).to(A.dtype).t(), bias)

        # 3. Save state
        ctx.state = state
        ctx.dtype_A, ctx.dtype_B, ctx.dtype_bias = A.dtype, B.dtype, None if bias is None else bias.dtype

        if any(ctx.needs_input_grad[:2]):
            ctx.tensors = (A, B)
        else:
            ctx.tensors = (None, None)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_empty:
            bias_grad = None if ctx.bias is None else torch.zeros_like(ctx.bias)
            return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, bias_grad, None

        req_gradA, _, _, req_gradBias, _= ctx.needs_input_grad
        A, B = ctx.tensors
        state = ctx.state

        grad_A, grad_B, grad_bias = None, None, None

        if req_gradBias:
            # compute grad_bias first before changing grad_output dtype
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)

        # not supported by PyTorch. TODO: create work-around
        #if req_gradB: grad_B = torch.matmul(grad_output.t(), A)
        if req_gradA: grad_A = torch.matmul(grad_output, F.dequantize_fp4(B, ctx.state).to(grad_output.dtype).t())

        return grad_A, grad_B, None, grad_bias, None
