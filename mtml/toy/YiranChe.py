__doc__ = "A decorator that takes a non-builtin function and adds the attribute __user_tag__ to it"

def add_user_tag(func):
    """ Add the attribute __user_tag__ to func.
	:params func: a non-builtin function to decorate
	:type func: function
	:rtype: function
    """
    func.__user_tag__ = "YiranChe"
    return func

