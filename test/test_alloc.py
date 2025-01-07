import sys
sys.path.append("..")

from src.utils.alloc import LRUAllocator
def test_alloc():
    allocator = LRUAllocator(["a", "b", "c", "d", "e"])

    allocator.get("1")
    allocator.get("2")
    allocator.get("3")
    allocator.get("4")
    allocator.get("5")
    allocator.get("1")
    allocator.get("2")
    assert allocator.get("6") == 'c', allocator.get("6")