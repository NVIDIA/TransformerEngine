def _gen_node_id() -> int:
    if "id" not in _gen_node_id.__dict__:
        _gen_node_id.id = 0
    _gen_node_id.id += 1
    return _gen_node_id.id


class Op:
    grad: "Op | None" = None

    def __init__(self):
        self.id = _gen_node_id()


class OpInputPlaceholder(Op):
    def replace(self, op: Op):
        self.__class__ = op.__class__
        self.__dict__ = op.__dict__
