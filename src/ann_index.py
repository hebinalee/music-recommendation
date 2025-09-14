import os
import json
import numpy as np
import faiss
from typing import Dict, Any, List, Tuple

class ANNIndex:
    def __init__(self, dim: int, index_path: str = './models/faiss_index.bin', mapping_path: str = './models/faiss_id_map.json'):
        self.dim = dim
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.index = None
        self.int_to_str: Dict[int, str] = {}
        self.str_to_int: Dict[str, int] = {}
    
    def build_from_embeddings(self, item_ids: List[str], embeddings: np.ndarray, use_ivf: bool = False, nlist: int = 1024) -> None:
        assert embeddings.shape[1] == self.dim
        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        emb = (embeddings / norms).astype('float32')
        id_map = faiss.IndexIDMap2(self._create_base_index(use_ivf, nlist))
        # create deterministic incremental ids and mapping
        self.int_to_str = {}
        self.str_to_int = {}
        int_ids = []
        for i, s in enumerate(item_ids):
            int_id = i
            self.int_to_str[int_id] = s
            self.str_to_int[s] = int_id
            int_ids.append(int_id)
        id_map.add_with_ids(emb, np.array(int_ids, dtype='int64'))
        self.index = id_map
    
    def _create_base_index(self, use_ivf: bool, nlist: int):
        if use_ivf:
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            return index
        return faiss.IndexFlatIP(self.dim)
    
    def train(self, embeddings: np.ndarray) -> None:
        if isinstance(self.index, faiss.IndexIVF):
            emb = embeddings.astype('float32')
            if not self.index.is_trained:
                self.index.train(emb)
    
    def save(self) -> None:
        if self.index is None:
            return
        faiss.write_index(self.index, self.index_path)
        with open(self.mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.int_to_str, f)
    
    def load(self) -> bool:
        if not os.path.exists(self.index_path):
            return False
        self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, 'r', encoding='utf-8') as f:
                self.int_to_str = {int(k): v for k, v in json.load(f).items()}
                self.str_to_int = {v: k for k, v in self.int_to_str.items()}
        return True
    
    def is_ready(self) -> bool:
        return self.index is not None
    
    def search(self, query_embeddings: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        assert self.index is not None
        # normalize
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-10
        q = (query_embeddings / norms).astype('float32')
        scores, ids = self.index.search(q, top_k)
        return scores, ids
    
    def decode_ids(self, ids: np.ndarray) -> List[str]:
        result: List[str] = []
        for row in ids:
            row_ids = []
            for x in row:
                row_ids.append(self.int_to_str.get(int(x), ''))
            result.append(row_ids)
        return result
