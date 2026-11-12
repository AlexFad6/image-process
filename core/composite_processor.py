from .image_processor_library import LibraryImageProcessor
from .image_processor_manual import ManualImageProcessor

class CompositeImageProcessor:
    """组合处理器：优先使用 library，缺失时回退到 manual。"""
    """Composite processor: prefer library, fallback to manual if missing."""

    def __init__(self, prefer="library"):
        if prefer == "library":
            self._primary = LibraryImageProcessor
            self._fallback = ManualImageProcessor
        else:
            self._primary = ManualImageProcessor
            self._fallback = LibraryImageProcessor

    def __getattr__(self, name):
        # 先尝试从首选处理器获取
        attr = getattr(self._primary, name, None)
        if attr is None:
            # 回退到备用处理器
            attr = getattr(self._fallback, name, None)
        if attr is None:
            raise AttributeError(
                f"Neither {self._primary.__name__} nor {self._fallback.__name__} "
                f"implements '{name}'"
            )
        return attr