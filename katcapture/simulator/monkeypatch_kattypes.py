from katcp.core import FailReply
import katcp.kattypes as kattypes
from katcp.kattypes import Parameter

def unpack_types(types, args, argnames):
    """Parse arguments according to types list.

    Parameters
    ----------
    types : list of kattypes
        The types of the arguments (in order).
    args : list of strings
        The arguments to parse.
    argnames : list of strings
        The names of the arguments.
    """
    try: multiple = types[-1].multiple
    except AttributeError: multiple = False

    if len(types) < len(args) and not multiple:
        raise FailReply("Too many parameters given.")

    # Wrap the types in parameter objects
    params = []
    for i, kattype in enumerate(types):
        name = ""
        if i < len(argnames):
            name = argnames[i]
        params.append(Parameter(i+1, name, kattype))


    if len(args) > len(types) and multiple:
        for i in range(len(types), len(args)):
            params.append(Parameter(i+1, name, kattype))

    # if len(args) < len(types) this passes in None for missing args
    return map(lambda param, arg: param.unpack(arg), params, args)

kattypes.unpack_types = unpack_types

class Str(kattypes.Str):
    def __init__(self, multiple=False, **kwargs):
        super(Str, self).__init__(**kwargs)
        self.multiple = multiple
        pass

kattypes.Str = Str
