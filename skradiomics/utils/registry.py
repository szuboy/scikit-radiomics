# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)

import inspect
import warnings
from collections import abc
from functools import partial


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.
    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.
    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


class Registry:
    """A registry to map strings to classes.

    Args:
        name (str): Registry name.
    """

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name=%s, items=%s)' % (self._name, self._module_dict)
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        """Get the registry record.

        Args:
            key (str): The func name in string format.

        Returns:
            function: The corresponding function.
        """
        return self._module_dict.get(key, None)

    def _register_module(self, module, module_name=None, force=False):
        if not (inspect.isfunction(module) or inspect.isclass(module)):
            raise TypeError('module must be a function or class, but got %s' % type(module))

        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        else:
            assert is_seq_of(module_name, str), ('module_name should be either of None, an instance of str or list, '
                                                 'but got %s' % type(module_name))
        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError('%s is already registered in %s' % (name, self.name))
            self._module_dict[name] = module

    def register_module(self, name=None, force=False, module=None):
        """Register a module.

        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Example:
            >>> extractor = Registry('extractor')
            >>> @extractor.register_module()
            >>> def square(image, mask, settings, **kwargs):
            >>>     pass

            >>> extractor = Registry('extractor')
            >>> @extractor.register_module(name='square')
            >>> def square(image, mask, settings, **kwargs):
            >>>     pass

        Args:
            name (str | None): The module name to be registered. If not specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with, the same name. Default: False.
            module (type): Module function to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError('force must be a boolean, but got %s' % type(force))

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):
            raise TypeError('name must be a str, but got %s' % type(name))

        # use it as a decorator: @x.register_module()
        def _register(callable_module):
            self._register_module(module=callable_module, module_name=name, force=force)
            return callable_module

        return _register

