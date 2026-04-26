import json
from torch.utils.data import Dataset
from typing import Dict, Any
from smartmap_mdv.utils import substitute_placeholders
from smartmap_mdv.config import DataConfig

def to_nmo_string(field_dict: Dict[str, Any],
                  mask_name: bool = False,
                  input_mode: str = "nmo",
                  drop_type: bool = False,
                  drop_path: bool = False,
                  no_placeholder: bool = False) -> str:
    """
    Converts a dictionary of field components into a canonical NMO string.

    This function centralizes the logic for serializing a field's attributes
    into a string format, supporting both 'nmo' (tagged) and 'msg' (raw) modes.
    """
    
    config = DataConfig(no_placeholder=no_placeholder)

    if input_mode == "msg":
        name_val = "" if mask_name else field_dict.get("name", "")
        type_val = field_dict.get("type", "")
        example_val = field_dict.get("example", "")
        path_val = field_dict.get("path", "")
        desc_val = substitute_placeholders(field_dict.get("desc", ""), config)
        
        parts = [name_val, type_val, example_val, path_val, desc_val]
        return " ".join(p for p in parts if p and p.strip())

    # NMO mode
    parts = []
    if not mask_name and field_dict.get("name"):
        parts.append(f"[NAME] {field_dict['name']}")
    if not drop_type and field_dict.get("type"):
        parts.append(f"[TYPE] {field_dict['type']}")
    if not drop_path and field_dict.get("path"):
        parts.append(f"[PATH] {field_dict['path']}")
    
    desc_val = substitute_placeholders(field_dict.get("desc", ""), config)
    if desc_val:
        parts.append(f"[DESC] {desc_val}")
    
    if field_dict.get("example"):
        parts.append(f"[EXAMPLE] {field_dict['example']}")
        
    return " ".join(parts)


class NMOFields:
    def __init__(self):
        self.fields: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def from_file(path: str, encoding: str = 'utf-8') -> 'NMOFields':
        inst = NMOFields()
        with open(path, 'r', encoding=encoding) as f:
            for line in f:
                field = json.loads(line)
                inst.fields[field["field_id"]] = field
        return inst
        
    def _get_field_as_dict(self, field_id: str) -> Dict[str, Any]:
        """Returns the field data as a standardized dictionary."""
        field = self.fields.get(field_id)
        if not field:
            return {}
            
        type_info = field.get("type", {})
        type_val = type_info.get("base", "") if isinstance(type_info, dict) else str(type_info)
            
        examples = field.get("examples", [])
        example_val = " ".join(examples) if examples else ""
        
        return {
            "name": field.get("name", ""),
            "type": type_val,
            "path": field.get("path", ""),
            "desc": field.get("description") or field.get("desc", ""),
            "example": example_val
        }

    def get_field_text(self, field_id: str, mask_name: bool = False, input_mode: str = "nmo", 
                     drop_type: bool = False, drop_path: bool = False, no_placeholder: bool = False) -> str:
        field_dict = self._get_field_as_dict(field_id)
        if not field_dict: 
            return ""
        return to_nmo_string(field_dict, mask_name, input_mode, drop_type, drop_path, no_placeholder)

    def get_all_texts(self, mask_name: bool = False, input_mode: str = "nmo", 
                    drop_type: bool = False, drop_path: bool = False, no_placeholder: bool = False) -> list[str]:
        return [self.get_field_text(fid, mask_name=mask_name, input_mode=input_mode, 
                                    drop_type=drop_type, drop_path=drop_path, 
                                    no_placeholder=no_placeholder) 
                for fid in self.fields.keys()]


def build_corpus(fieldsA_path, fieldsB_path, mask_name=False, input_mode="nmo", drop_type=False, drop_path=False, no_placeholder=False):
    fieldsA = NMOFields.from_file(fieldsA_path)
    fieldsB = NMOFields.from_file(fieldsB_path)
    corpusA = [(fid, fieldsA.get_field_text(fid, mask_name=mask_name, input_mode=input_mode, 
                                            drop_type=drop_type, drop_path=drop_path, 
                                            no_placeholder=no_placeholder)) 
               for fid in fieldsA.fields.keys()]
    corpusB = [(fid, fieldsB.get_field_text(fid, mask_name=mask_name, input_mode=input_mode, 
                                            drop_type=drop_type, drop_path=drop_path, 
                                            no_placeholder=no_placeholder)) 
               for fid in fieldsB.fields.keys()]
    return fieldsA, fieldsB, corpusA, corpusB


def load_pairs(pairs_path):
    pairs = []
    with open(pairs_path, 'r', encoding='utf-8') as f:
        for line in f:
            p = json.loads(line)
            if p.get("label", 1) == 1:
                pairs.append((p["source"], p["target"]))
    return pairs


class PairDataset(Dataset):
    def __init__(self, fieldsA_path, fieldsB_path, pairs_path, max_pairs=100000, input_mode="nmo"):
        fieldsA = NMOFields.from_file(fieldsA_path)
        fieldsB = NMOFields.from_file(fieldsB_path)
        self.a_texts, self.b_texts = [], []
        with open(pairs_path) as f:
            for line in f:
                if len(self.a_texts) >= max_pairs: 
                    break
                p = json.loads(line)
                self.a_texts.append(fieldsA.get_field_text(p["source"], input_mode=input_mode))
                self.b_texts.append(fieldsB.get_field_text(p["target"], input_mode=input_mode))
        print(f"Complete loading PairDataset.  {len(self.a_texts)} paired datasets. (mode: {input_mode})")

    def __len__(self):
        return len(self.a_texts)

    def __getitem__(self, idx):
        return self.a_texts[idx], self.b_texts[idx]
