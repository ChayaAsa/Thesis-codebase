# Model file I/O

Read, write, validate, and template motor-model YAML files. The on-disk
text is the source of truth — value updates use line-level patching so
inline comments and indentation are preserved.

```python
from motortools import (
    list_models, read_model_text, write_model_text, delete_model_file,
    parse_model_values, patch_yaml_value,
    validate_model_text, template_text,
    EditableModel,
)
```

The 12 required fields in every model YAML (also exported as
`motortools.models_io.REQUIRED_FIELDS`):

```
p_min, p_max, v_min, v_max, t_min, t_max,
kp_min, kp_max, kd_min, kd_max,
gear_ratio, kt, pole_pairs, max_temp
```

---

## Listing and reading

| Function | Description |
|---|---|
| `list_models() -> list[str]` | Sorted built-in model names (filename stems, e.g. `'ak45-10'`). |
| `read_model_text(name) -> str` | Raw YAML text for a built-in model. Raises `FileNotFoundError`. |
| `parse_model_values(text) -> dict` | YAML text → `{key: value}`. Empty/whitespace input returns `{}`. Raises `yaml.YAMLError` or `ValueError` on malformed input. |

---

## Writing and deleting

`write_model_text` and `delete_model_file` make a `.bak` copy by default
so the previous content survives every edit.

| Function | Description |
|---|---|
| `write_model_text(name, text, *, backup=True) -> Path` | Write `text` to `<name>.yaml`. With `backup=True`, copies the previous file to `<name>.yaml.bak` first. Returns the written path. |
| `delete_model_file(name, *, backup=True) -> Path` | Delete `<name>.yaml`. With `backup=True`, copies it to `<name>.yaml.bak` first. Raises `FileNotFoundError` if the YAML does not exist. |

---

## Editing and validation

| Function | Description |
|---|---|
| `patch_yaml_value(text, key, value) -> str` | Replace the value of `key` in `text` without touching comments, indentation, or other lines. Appends `key: value` if not present. Floats are formatted with up to 6 significant digits; ints stay ints. |
| `validate_model_text(text) -> list[str]` | Return human-readable issues. Empty list = OK. Checks: parse error, missing required keys, type errors, `*_min < *_max`, positive-only fields > 0. Mirrors the rules in `tmotorcan.models`. |
| `template_text(name) -> str` | Annotated YAML skeleton for a brand-new model, with sensible AK-series defaults. |

```python
from motortools import read_model_text, patch_yaml_value, validate_model_text

text = read_model_text('ak45-10')
text = patch_yaml_value(text, 'max_temp', 85)   # comments preserved
issues = validate_model_text(text)
if not issues:
    print(text)
```

---

## `EditableModel`

Mutable wrapper around a model YAML. Each required field is exposed as an
attribute; the setter rewrites the underlying YAML text via
`patch_yaml_value`, so comments and indentation survive every edit.
Numeric type is enforced (int for `pole_pairs` / `max_temp`; int-or-float
for the rest).

### Construction

| Constructor | Description |
|---|---|
| `EditableModel(name, text)` | Direct constructor. |
| `EditableModel.load(name)` | Load an existing built-in model. |
| `EditableModel.new(name)` | Start a fresh model from `template_text(name)` (not yet saved). |

### Properties and methods

| Member | Description |
|---|---|
| `m.name` | Read-only model name. |
| `m.text` | Read-only current YAML text. |
| `m.replace_text(value)` | Swap the entire YAML text (e.g. after editing in a TextArea). |
| `m.<field>` (e.g. `m.p_min`) | Read the parsed value for any required field. Returns `None` if missing or YAML doesn't parse. |
| `m.<field> = value` | Patch the YAML text in place. Raises `TypeError` on wrong numeric type, `AttributeError` on unknown fields. |
| `m.values() -> dict` | Parse the current YAML. Returns `{}` on parse error. |
| `m.validate() -> list[str]` | List of validation issues (empty = OK). |
| `m.save(*, backup=True, force=False)` | Write to `<name>.yaml` (with `.bak` backup). Raises `ValueError` on validation issues unless `force=True`. |

### Example

```python
from motortools import EditableModel

m = EditableModel.load('ak45-10')
m.p_min = -15.0
m.p_max =  15.0
m.max_temp = 85

issues = m.validate()
if issues:
    for line in issues:
        print('  -', line)
else:
    m.save()                 # writes ak45-10.yaml + ak45-10.yaml.bak

# brand-new model
new = EditableModel.new('my-motor')
new.kt = 0.12
new.gear_ratio = 6.0
new.save()
```
