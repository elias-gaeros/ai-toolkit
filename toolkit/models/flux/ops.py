import torch



def cast_to_input(
    weight: torch.Tensor, input: torch.Tensor, non_blocking: bool = False
) -> torch.Tensor:
    """
    Casts the given weight tensor to the same dtype and device as the input tensor.

    Args:
        weight (torch.Tensor): The weight tensor to be cast.
        input (torch.Tensor): The input tensor to determine the dtype and device.
        non_blocking (bool, optional): If True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.

    Returns:
        torch.Tensor: The casted weight tensor.
    """
    return weight.to(device=input.device, dtype=input.dtype, non_blocking=non_blocking)


def attention_pytorch(
    q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False
):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
    )
    out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


def cast_bias_weight(s, input=None, dtype=None, device=None):
    if input is not None:
        if dtype is None:
            dtype = input.dtype
        if device is None:
            device = input.device

    bias = None
    non_blocking = False
    if s.bias is not None:
        bias = s.bias.to(device=device, dtype=dtype, non_blocking=non_blocking)
    weight = s.weight.to(device=device, dtype=dtype, non_blocking=non_blocking)
    return weight, bias


class nn:
    Linear = torch.nn.Linear
    # GroupNorm = torch.nn.GroupNorm
    LayerNorm = torch.nn.LayerNorm

    class Embedding(torch.nn.Embedding):
        def forward(self, input, out_dtype=None):
            output_dtype = out_dtype
            if (
                self.weight.dtype == torch.float16
                or self.weight.dtype == torch.bfloat16
            ):
                out_dtype = None
            return torch.nn.functional.embedding(
                input,
                self.weight.to(dtype=output_dtype, device=input.device),
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )  # .to(dtype=output_dtype)


class manual_cast:
    class Linear(torch.nn.Linear):

        def forward(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

    # class GroupNorm(torch.nn.GroupNorm):

    #     def forward(self, input):
    #         weight, bias = cast_bias_weight(self, input)
    #         return torch.nn.functional.group_norm(
    #             input, self.num_groups, weight, bias, self.eps
    #         )

    class LayerNorm(torch.nn.LayerNorm):
        def reset_parameters(self):
            return None

        def forward(self, input):
            if self.weight is not None:
                weight, bias = cast_bias_weight(self, input)
            else:
                weight = None
                bias = None
            return torch.nn.functional.layer_norm(
                input, self.normalized_shape, weight, bias, self.eps
            )

    class Embedding(torch.nn.Embedding):
        def forward(self, input, out_dtype=None):
            output_dtype = out_dtype
            if (
                self.weight.dtype == torch.float16
                or self.weight.dtype == torch.bfloat16
            ):
                out_dtype = None
            return torch.nn.functional.embedding(
                input,
                self.weight.to(dtype=output_dtype, device=input.device),
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )  # .to(dtype=output_dtype)


def get_ops_namespace(dtype, device):
    if dtype in {torch.float16, torch.bfloat16, torch.float32}:
        return nn
    else:
        return manual_cast