from models.dual_expert import DualExpert


def get_model(model_name, args):
    name = model_name.lower()
    if name == "dual_expert":
        return DualExpert(args)
    else:
        assert 0
