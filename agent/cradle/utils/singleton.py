"""A singleton metaclass for ensuring only one instance of a class."""
import abc
import threading


class Singleton(abc.ABCMeta, type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    """

    _instances = {}
    _lock = threading.RLock()

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        instance = cls._instances[cls]
        if (args or kwargs) and hasattr(instance, "_singleton_reconfigure"):
            instance._singleton_reconfigure(*args, **kwargs)
        return instance


class AbstractSingleton(abc.ABC, metaclass=Singleton):
    """
    Abstract singleton class for ensuring only one instance of a class.
    """

    pass
