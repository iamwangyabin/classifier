import warnings

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. ")


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
