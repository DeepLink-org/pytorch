from torch._dynamo import register_backend


@register_backend
def topsgraph(gm, fake_input_tensor):
    from third_party.DICP.common.compile_fx import compile_fx

    return compile_fx(gm, fake_input_tensor, "topsgraph")
