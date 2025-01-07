from typing import List, Dict


class LRUAllocator:
    def __init__(self, candidates: List[str], allocated: Dict[str, str] = {}):
        self.candidates = candidates
        self.allocated = allocated
        self.free = [v for v in candidates if v not in allocated]
        self.tracks = []

    def get(self, key: str):
        if key not in self.allocated:
            self._alloc(key)
        if key in self.tracks:
            self.tracks.remove(key)
        self.tracks.insert(0, key)
        return self.allocated[key]

    def _alloc(self, key: str):
        if len(self.allocated) >= len(self.candidates):
            value = self.allocated[self.tracks.pop()]
        else:
            value = self.free.pop()
        self.allocated[key] = value
