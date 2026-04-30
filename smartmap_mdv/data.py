import json
import re
from typing import Dict, Any, List, Tuple
from torch.utils.data import Dataset

from smartmap_mdv.utils import substitute_placeholders
from smartmap_mdv.config import DataConfig


def _split_identifier(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _format_path_text(path: str) -> str:
    path = str(path or "").strip()
    if not path:
        return ""
    parts = [p for p in path.split(".") if p]
    if not parts:
        return path
    parts = [_split_identifier(p) for p in parts]
    return " . ".join(p for p in parts if p)


def serialize_field_text(
    field: Dict[str, Any],
    style: str = "flat_field",
    include_type: bool = True,
    include_path: bool = True,
    include_desc: bool = False,
    include_example: bool = False,
) -> str:
    name = _split_identifier(field.get("name", ""))
    type_val = str(field.get("type", "") or "").strip()
    path = _format_path_text(field.get("path", ""))
    desc = str(field.get("desc") or field.get("description") or "").strip()
    example = field.get("example", "")
    if not example and field.get("examples"):
        exs = field.get("examples")
        if isinstance(exs, list):
            example = " ".join(str(x) for x in exs if x)
        else:
            example = str(exs or "")
    example = str(example or "").strip()

    parts = []
    if name:
        parts.append(name)
    if include_type and type_val:
        parts.append(type_val)
    if include_path and path:
        parts.append(path)
    if include_desc and desc:
        parts.append(desc)
    if include_example and example:
        parts.append(example)

    if style == "nmo":
        tagged = []
        if name:
            tagged.append(f"[NAME] {name}")
        if include_type and type_val:
            tagged.append(f"[TYPE] {type_val}")
        if include_path and path:
            tagged.append(f"[PATH] {path}")
        return " ".join(tagged).strip()

    return " ".join(parts).strip()


def to_nmo_string(
    field_dict: Dict[str, Any],
    mask_name: bool = False,
    input_mode: str = "nmo",
    drop_type: bool = False,
    drop_path: bool = False,
    drop_desc: bool = False,
    drop_example: bool = False,
    no_placeholder: bool = False,
) -> str:
    """
    Serialize a field dictionary into one of:
      - raw_msg: raw message/context only
      - flat_field: flattened text without tags
      - nmo: tagged NMO-style text using only NAME, TYPE, and PATH
    """
    config = DataConfig(no_placeholder=no_placeholder)

    raw_msg = (field_dict.get("raw_msg") or field_dict.get("msg") or "").strip()
    name_val = "" if mask_name else (field_dict.get("name", "") or "").strip()
    type_val = (field_dict.get("type", "") or "").strip()
    path_val = (field_dict.get("path", "") or "").strip()
    desc_val = (field_dict.get("desc", "") or "").strip()
    example_val = (field_dict.get("example", "") or "").strip()

    if not no_placeholder:
        desc_val = substitute_placeholders(desc_val, config)
        example_val = substitute_placeholders(example_val, config)

    if input_mode == "raw_msg":
        return raw_msg

    field_view = {
        "name": name_val,
        "type": type_val,
        "path": path_val,
        "desc": desc_val,
        "example": example_val,
    }

    if input_mode == "flat_field":
        return serialize_field_text(
            field_view,
            style="flat_field",
            include_type=not drop_type,
            include_path=not drop_path,
            include_desc=not drop_desc,
            include_example=not drop_example,
        )

    # default: nmo
    return serialize_field_text(
        field_view,
        style="nmo",
        include_type=not drop_type,
        include_path=not drop_path,
        include_desc=False,
        include_example=False,
    )


class NMOFields:
    def __init__(self):
        self.fields: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def from_file(path: str, encoding: str = "utf-8") -> "NMOFields":
        inst = NMOFields()
        with open(path, "r", encoding=encoding) as f:
            for line in f:
                if not line.strip():
                    continue
                field = json.loads(line)
                inst.fields[field["field_id"]] = field
        return inst

    def _get_field_as_dict(self, field_id: str) -> Dict[str, Any]:
        """Return field data as a normalized dictionary."""
        field = self.fields.get(field_id)
        if not field:
            return {}

        type_info = field.get("type", {})
        type_val = type_info.get("base", "") if isinstance(type_info, dict) else str(type_info or "")

        examples = field.get("examples", [])
        if isinstance(examples, list):
            example_val = " ".join(str(x) for x in examples if x)
        else:
            example_val = str(examples or "")

        return {
            "field_id": field_id,
            "name": field.get("name", "") or "",
            "type": type_val,
            "path": field.get("path", "") or "",
            "desc": field.get("description") or field.get("desc", "") or "",
            "example": example_val,
            "raw_msg": field.get("raw_msg", "") or field.get("msg", "") or "",
            "vendor": field.get("vendor") or field.get("vendor_id") or "",
            "product_family": field.get("product_family") or field.get("product") or "",
        }

    def get_field_text(
        self,
        field_id: str,
        mask_name: bool = False,
        input_mode: str = "nmo",
        drop_type: bool = False,
        drop_path: bool = False,
        drop_desc: bool = False,
        drop_example: bool = False,
        no_placeholder: bool = False,
    ) -> str:
        field_dict = self._get_field_as_dict(field_id)
        if not field_dict:
            return ""
        return to_nmo_string(
            field_dict,
            mask_name=mask_name,
            input_mode=input_mode,
            drop_type=drop_type,
            drop_path=drop_path,
            drop_desc=drop_desc,
            drop_example=drop_example,
            no_placeholder=no_placeholder,
        )

    def get_all_texts(
        self,
        mask_name: bool = False,
        input_mode: str = "nmo",
        drop_type: bool = False,
        drop_path: bool = False,
        drop_desc: bool = False,
        drop_example: bool = False,
        no_placeholder: bool = False,
    ) -> List[str]:
        return [
            self.get_field_text(
                fid,
                mask_name=mask_name,
                input_mode=input_mode,
                drop_type=drop_type,
                drop_path=drop_path,
                drop_desc=drop_desc,
                drop_example=drop_example,
                no_placeholder=no_placeholder,
            )
            for fid in self.fields.keys()
        ]


def build_corpus(
    fieldsA_path: str,
    fieldsB_path: str,
    mask_name: bool = False,
    input_mode: str = "nmo",
    drop_type: bool = False,
    drop_path: bool = False,
    drop_desc: bool = False,
    drop_example: bool = False,
    no_placeholder: bool = False,
):
    fieldsA = NMOFields.from_file(fieldsA_path)
    fieldsB = NMOFields.from_file(fieldsB_path)

    corpusA = [
        (
            fid,
            fieldsA.get_field_text(
                fid,
                mask_name=mask_name,
                input_mode=input_mode,
                drop_type=drop_type,
                drop_path=drop_path,
                drop_desc=drop_desc,
                drop_example=drop_example,
                no_placeholder=no_placeholder,
            ),
        )
        for fid in fieldsA.fields.keys()
    ]

    corpusB = [
        (
            fid,
            fieldsB.get_field_text(
                fid,
                mask_name=mask_name,
                input_mode=input_mode,
                drop_type=drop_type,
                drop_path=drop_path,
                drop_desc=drop_desc,
                drop_example=drop_example,
                no_placeholder=no_placeholder,
            ),
        )
        for fid in fieldsB.fields.keys()
    ]

    return fieldsA, fieldsB, corpusA, corpusB


def load_pairs(pairs_path: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            if p.get("label", 1) == 1:
                pairs.append((p["source"], p["target"]))
    return pairs


class PairDataset(Dataset):
    def __init__(
        self,
        fieldsA_path: str,
        fieldsB_path: str,
        pairs_path: str,
        max_pairs: int = 100000,
        input_mode: str = "nmo",
        mask_name: bool = False,
        drop_type: bool = False,
        drop_path: bool = False,
        drop_desc: bool = False,
        drop_example: bool = False,
        no_placeholder: bool = False,
        encoding: str = "utf-8",
    ):
        fieldsA = NMOFields.from_file(fieldsA_path, encoding=encoding)
        fieldsB = NMOFields.from_file(fieldsB_path, encoding=encoding)

        self.a_texts: List[str] = []
        self.b_texts: List[str] = []

        with open(pairs_path, "r", encoding=encoding) as f:
            for line in f:
                if len(self.a_texts) >= max_pairs:
                    break
                if not line.strip():
                    continue

                p = json.loads(line)
                if p.get("label", 1) != 1:
                    continue

                self.a_texts.append(
                    fieldsA.get_field_text(
                        p["source"],
                        mask_name=mask_name,
                        input_mode=input_mode,
                        drop_type=drop_type,
                        drop_path=drop_path,
                        drop_desc=drop_desc,
                        drop_example=drop_example,
                        no_placeholder=no_placeholder,
                    )
                )
                self.b_texts.append(
                    fieldsB.get_field_text(
                        p["target"],
                        mask_name=mask_name,
                        input_mode=input_mode,
                        drop_type=drop_type,
                        drop_path=drop_path,
                        drop_desc=drop_desc,
                        drop_example=drop_example,
                        no_placeholder=no_placeholder,
                    )
                )

        print(
            f"Complete loading PairDataset. {len(self.a_texts)} paired datasets. "
            f"(mode: {input_mode})"
        )

    def __len__(self):
        return len(self.a_texts)

    def __getitem__(self, idx):
        return self.a_texts[idx], self.b_texts[idx]
