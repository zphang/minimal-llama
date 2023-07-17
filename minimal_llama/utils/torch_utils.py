def get_requires_grad(model):
    requires_grad_params = [n for n, p in model.named_parameters() if p.requires_grad]
    state_dict = model.state_dict()
    return {k: state_dict[k] for k in requires_grad_params}
