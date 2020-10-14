__doc__ = "Decorator function that takes a non-builtin Python function and adds the __user_tag__ attribute."

def add_user_tag(f):
	'''Adds np788 username as an attribute ``__user_tag__`` to input function ``f``.
    
    :param f: Function to decorate
    :type f: function
    :rtype: function
	'''
	f.__user_tag__ = 'np788'
	return f