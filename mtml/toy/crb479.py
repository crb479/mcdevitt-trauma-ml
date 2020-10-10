# For Demo Task
__doc__ = "Decorator fxn to add '__user_tag__' to non-built-in Python fxn"

def add_user_tag(fxn):
	'''Adds username as an attribute ``__user_tag__`` to ``fxn``.
    
    :param fxn: Function to decorate
    :type fxn: function
    :rtype: function
	'''
	fxn.__user_tag__ = 'crb479'
	return fxn
