__doc__ = "Decorator adding ``__user_tag__`` to a non-builtin Python function."

def add_user_tag(f):
    """Adds my GitHub username as an attribute ``__user_tag__`` to ``f``.
    
    :param f: Function to decorate
    :type f: function
    :rtype: function
    """
    f.__user_tag__ = "phetdam"
    return f