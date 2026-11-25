"""The mounting sheet of an OpenSci purchase as text, in English or Simplified Chinese.

A mounting sheet comes with OpenSci mounts chosen per optic without a laboratory layout (a part list with mechanical
supply ``DERIVED``, or an ``OPTICAL_SYSTEM`` design bought with derived mounts and no platform). The service delivers it
as JSON (``OpenSciClient.resolution_mounting_sheet`` / ``order_mounting_sheet``) with English and Simplified Chinese
text; ``render_mounting_sheet(document, locale)`` writes it for people: per optic the number of sets, the beam height
and angle of incidence (and where that angle comes from), the mount / post / post holder of each set, the assembly
steps, the beam-height arithmetic, the adjustments the hardware provides, the checks made, the checks not made with
their reason, the assumed values and the conditions; then the sheet's notes.

Only a check whose ``status`` is ``PASS`` is listed under the checks made; every other check is listed as not made.
A text field is read in the requested language: ``<field>_zh_cn`` (or ``{"en": ..., "zh-CN": ...}``) for ``zh-CN``,
``<field>`` for English, each falling back to the other. Fields the document does not carry are left out, and fields
it adds are ignored.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

LOCALES = ("en", "zh-CN")
#: The headings of the text, per language.
LABELS = {
    "en": {"title": "Mounting sheet", "resolution": "Resolution", "order": "Order", "unit": "Optic", "line": "line",
           "instance": "component", "sets": "Sets", "beam_height": "Beam height", "incidence": "Angle of incidence",
           "parts": "Parts of each set", "steps": "Assembly", "height": "Beam height arithmetic",
           "axes": "Adjustments", "no_axes": "none (fixed)", "checks": "Checks made",
           "not_checked": "Not checked", "assumptions": "Assumed values", "conditions": "Conditions",
           "total": "Total", "notes": "Notes", "status": "status"},
    "zh-CN": {"title": "安装说明", "resolution": "采购配置", "order": "订单号", "unit": "光学元件", "line": "行",
              "instance": "元件", "sets": "套数", "beam_height": "光轴高度", "incidence": "入射角",
              "parts": "每套零件", "steps": "安装步骤", "height": "光轴高度计算", "axes": "调节",
              "no_axes": "无（固定）", "checks": "已做的检查", "not_checked": "未做的检查", "assumptions": "假定值",
              "conditions": "使用条件", "total": "合计", "notes": "说明", "status": "状态"},
}
#: The words of the part roles of a set (an unknown role is written from its code).
ROLE_WORDS = {
    "en": {"MOUNT": "Mount", "POST": "Post", "POST_HOLDER": "Post holder", "ADAPTER": "Adapter",
           "POST_HOLDER_BASE": "Post holder base", "BASE": "Base", "CLAMP": "Clamp"},
    "zh-CN": {"MOUNT": "镜架", "POST": "支杆", "POST_HOLDER": "支杆座", "ADAPTER": "转接件",
              "POST_HOLDER_BASE": "支杆座底座", "BASE": "底座", "CLAMP": "压板"},
}
#: Where the angle of incidence comes from (``incidence_basis``).
INCIDENCE_BASIS_WORDS = {
    "en": {"STATED": "as you stated", "DESIGN": "from your design",
           "DEFAULT_FOR_CATEGORY": "the default for this kind of optic; state incidence_deg to change it"},
    "zh-CN": {"STATED": "按您给出的值", "DESIGN": "取自您的设计",
              "DEFAULT_FOR_CATEGORY": "按此类光学元件的默认值；如需改变请给出 incidence_deg"},
}
#: The words of the adjustment mechanisms, for an adjustment the sheet gives without its own text.
MECHANISM_WORDS = {
    "en": {"TIP_TILT_MOUNT": "the mount's adjusters", "MOUNT_TILT_HORIZONTAL_AXIS": "the mount's adjuster",
           "MOUNT_TILT_VERTICAL_AXIS": "the mount's adjuster", "POST_HEIGHT_CLAMP": "slide the post in its holder",
           "POST_HEIGHT": "slide the post in its holder", "BASE_SLOT_TRAVEL": "move the holder along its base slot",
           "BASE_SLOT": "move the holder along its base slot"},
    "zh-CN": {"TIP_TILT_MOUNT": "镜架调节旋钮", "MOUNT_TILT_HORIZONTAL_AXIS": "镜架调节旋钮",
              "MOUNT_TILT_VERTICAL_AXIS": "镜架调节旋钮", "POST_HEIGHT_CLAMP": "在支杆座中上下滑动支杆",
              "POST_HEIGHT": "在支杆座中上下滑动支杆", "BASE_SLOT_TRAVEL": "沿底座槽移动支杆座",
              "BASE_SLOT": "沿底座槽移动支杆座"},
}
#: The words of a check status other than PASS / NOT_RUN.
STATUS_WORDS = {"en": {"FAIL": "did not pass"}, "zh-CN": {"FAIL": "未通过"}}
#: Keys a unit list may be given under, and the beam-height arithmetic of a unit.
UNIT_KEYS = ("units", "optics")
HEIGHT_KEYS = ("height_arithmetic", "height")
PASS = "PASS"
NOT_RUN = "NOT_RUN"
#: The code points a terminal shows two columns wide (Hangul Jamo, CJK, Hangul syllables, compatibility ideographs,
#: vertical / small forms, full-width forms), for the underline of a title.
WIDE_RANGES = ((0x1100, 0x115F), (0x2E80, 0xA4CF), (0xAC00, 0xD7A3), (0xF900, 0xFAFF), (0xFE30, 0xFE4F),
               (0xFF00, 0xFF60), (0xFFE0, 0xFFE6))
WIDE_COLUMNS = 2


def _width(text: str) -> int:
    return sum(WIDE_COLUMNS if any(low <= ord(ch) <= high for low, high in WIDE_RANGES) else 1 for ch in text)


def _number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return f"{value:g}"


def _code_words(code: Any) -> str:
    return str(code).replace("_", " ").capitalize()


def _text(item: Any, field: str, locale: str) -> str | None:
    """The text of ``field`` of ``item`` in ``locale`` (falling back to the other language), or None."""

    if not isinstance(item, Mapping):
        return None
    value = item.get(field)
    if isinstance(value, Mapping):
        chosen = value.get(locale) or value.get("zh-CN" if locale == "en" else "en")
        return str(chosen) if chosen else None
    localized = item.get(f"{field}_zh_cn")
    first, second = (localized, value) if locale == "zh-CN" else (value, localized)
    chosen = first if first not in (None, "") else second
    return str(chosen) if chosen not in (None, "") else None


def _name(item: Any, locale: str) -> str:
    sku = item.get("opensci_sku") if isinstance(item, Mapping) else None
    name = _text(item, "display_name", locale)
    return " ".join(part for part in (str(sku) if sku else None, f"({name})" if name and sku else name) if part)


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _texts(items: Any, locale: str) -> list[str]:
    out = []
    for item in _list(items):
        text = _text(item, "text", locale) if isinstance(item, Mapping) else (str(item) if item else None)
        if text is None and isinstance(item, Mapping) and item.get("code"):
            text = _code_words(item["code"])
        if text:
            out.append(text)
    return out


def _part_count(total: Any, sets: Any, locale: str) -> str | None:
    """The count of a part in each set. A chain line's ``quantity`` is the line total (every set together), so it is
    divided by the number of sets; ``× 1 (2 in total)`` / ``× 1（共 2 件）`` when there is more than one set. A total
    the sets do not divide evenly is written as the total."""

    if total is None:
        return None
    numeric = (isinstance(total, (int, float)) and isinstance(sets, (int, float)) and not isinstance(total, bool)
               and not isinstance(sets, bool) and sets > 0)
    if not numeric or sets == 1:
        return f"× {_number(total)}"
    per_set, remainder = divmod(total, sets)
    if remainder:
        return f"共 {_number(total)} 件" if locale == "zh-CN" else f"{_number(total)} in total"
    return (f"× {_number(per_set)}（共 {_number(total)} 件）" if locale == "zh-CN"
            else f"× {_number(per_set)} ({_number(total)} in total)")


def _sets(count: Any, locale: str) -> str:
    if locale == "zh-CN":
        return f"{_number(count)} 套"
    return f"{_number(count)} set" + ("" if count == 1 else "s")


def _adjustment(item: Any, locale: str) -> str | None:
    if not isinstance(item, Mapping):
        return str(item) if item else None
    text = _text(item, "text", locale)
    if text:
        return text
    axis = item.get("axis")
    mechanism = item.get("mechanism_code")
    words = MECHANISM_WORDS[locale].get(mechanism) or (_code_words(mechanism) if mechanism else LABELS[locale]["no_axes"])
    travel = item.get("travel") if isinstance(item.get("travel"), Mapping) else None
    span = (f", {_number(travel.get('minus'))} / +{_number(travel.get('plus'))} {travel.get('unit', '')}".rstrip()
            if travel else "")
    return f"{axis}: {words}{span}" if axis else None


def _adjustment_lines(unit: Mapping[str, Any], labels: Mapping[str, str], locale: str) -> list[str]:
    if "adjustments" in unit and _list(unit.get("adjustments")):
        texts = [t for t in (_adjustment(a, locale) for a in _list(unit["adjustments"])) if t]
    elif "adjustment_axes" in unit or "adjustments" in unit:
        axes = _list(unit.get("adjustment_axes_zh_cn")) if locale == "zh-CN" else []
        texts = [str(a) for a in (axes or _list(unit.get("adjustment_axes")))]
    else:
        return []
    if not texts:
        return [f"   {labels['axes']}: {labels['no_axes']}"]
    return [f"   {labels['axes']}:", *(f"     - {t}" for t in texts)]


def _check_lines(unit: Mapping[str, Any], labels: Mapping[str, str], locale: str) -> list[str]:
    made: list[str] = []
    not_made: list[str] = []
    for check in _list(unit.get("checks")):
        if not isinstance(check, Mapping):
            if check:
                not_made.append(str(check))  # a check without a status is not stated as made
            continue
        text = _text(check, "text", locale) or (_code_words(check["code"]) if check.get("code") else None)
        if not text:
            continue
        status = check.get("status")
        if status == PASS:
            made.append(text)
            continue
        reason = _text(check, "reason", locale)
        if reason and reason != text:
            text = f"{text} ({reason})"
        if status not in (NOT_RUN, None):
            word = STATUS_WORDS[locale].get(str(status)) or _code_words(status).lower()
            text = f"{text} ({labels['status']}: {word})"
        not_made.append(text)
    lines = []
    for label, texts in (("checks", made), ("not_checked", not_made)):
        if texts:
            lines.append(f"   {labels[label]}:")
            lines += [f"     - {t}" for t in texts]
    return lines


def _unit_lines(unit: Mapping[str, Any], index: int, labels: Mapping[str, str], locale: str) -> list[str]:
    heading = f"{index}. {labels['unit']}: {_name(unit.get('optic') or {}, locale)}".rstrip(": ")
    where = []
    if unit.get("line_id"):
        where.append(f"{labels['line']} {unit['line_id']}")
    if unit.get("instance_id"):
        where.append(f"{labels['instance']} {unit['instance_id']}")
    if unit.get("unit_id") and unit["unit_id"] not in (unit.get("line_id"), unit.get("instance_id")):
        where.append(str(unit["unit_id"]))
    lines = [heading + (f"  [{', '.join(where)}]" if where else "")]
    if unit.get("quantity") is not None:
        lines.append(f"   {labels['sets']}: {_sets(unit['quantity'], locale)}")
    if unit.get("beam_height_mm") is not None:
        lines.append(f"   {labels['beam_height']}: {_number(unit['beam_height_mm'])} mm")
    if unit.get("incidence_deg") is not None:
        basis = INCIDENCE_BASIS_WORDS[locale].get(unit.get("incidence_basis"))
        lines.append(f"   {labels['incidence']}: {_number(unit['incidence_deg'])}°" + (f" ({basis})" if basis else ""))
    parts = _list(unit.get("chain"))
    if parts:
        lines.append(f"   {labels['parts']}:")
        for part in parts:
            if not isinstance(part, Mapping):
                continue
            role = part.get("role")
            role_text = ROLE_WORDS[locale].get(role) or ROLE_WORDS["en"].get(role) or (_code_words(role) if role else None)
            lines.append("     - " + " ".join(p for p in (f"{role_text}:" if role_text else None, _name(part, locale),
                                                         _part_count(part.get("quantity"), unit.get("quantity"), locale))
                                                if p))
    steps = [_text(step, "text", locale) if isinstance(step, Mapping) else str(step) for step in _list(unit.get("steps"))]
    if any(steps):
        lines.append(f"   {labels['steps']}:")
        lines += [f"     {n}. {step}" for n, step in enumerate((s for s in steps if s), start=1)]
    height = next((unit[key] for key in HEIGHT_KEYS if isinstance(unit.get(key), Mapping)), None)
    if height is not None:
        lines.append(f"   {labels['height']}:")
        summary = _text(height, "text", locale)
        if summary:
            lines.append(f"     {summary}")
        else:
            for term in _list(height.get("terms")):
                label = _text(term, "label", locale) or ""
                value = term.get("mm") if isinstance(term, Mapping) else None
                lines.append(f"     {label}: {_number(value)} mm" if value is not None else f"     {label}")
            if height.get("total_mm") is not None:
                lines.append(f"     {labels['total']}: {_number(height['total_mm'])} mm")
    lines += _adjustment_lines(unit, labels, locale)
    lines += _check_lines(unit, labels, locale)
    for key in ("assumptions", "conditions"):
        texts = _texts(unit.get(key), locale)
        if texts:
            lines.append(f"   {labels[key]}:")
            lines += [f"     - {t}" for t in texts]
    return lines


def render_mounting_sheet(document: Mapping[str, Any], locale: str = "en") -> str:
    """The mounting sheet ``document`` (the service's JSON) as text in ``locale`` (``en`` or ``zh-CN``)."""

    if locale not in LOCALES:
        raise ValueError(f"locale must be one of {', '.join(LOCALES)}")
    if not isinstance(document, Mapping):
        raise ValueError("a mounting sheet is a JSON object")
    labels = LABELS[locale]
    title = _text(document, "title", locale) or labels["title"]
    lines = [title, "=" * _width(title)]
    for key, label in (("order_number", "order"), ("resolution_id", "resolution")):
        if document.get(key):
            lines.append(f"{labels[label]}: {document[key]}")
    units = next((_list(document[key]) for key in UNIT_KEYS if key in document), [])
    for index, unit in enumerate((u for u in units if isinstance(u, Mapping)), start=1):
        lines.append("")
        lines += _unit_lines(unit, index, labels, locale)
    notes = _texts(document.get("notes"), locale)
    if notes:
        lines += ["", f"{labels['notes']}:", *(f"- {note}" for note in notes)]
    return "\n".join(lines) + "\n"


__all__ = ["render_mounting_sheet"]
