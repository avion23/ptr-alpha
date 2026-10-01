"""Member identity and partial-insert validation for OCR rows."""

from scripts.gemini_ocr_common import validate_transactions


def _tx(asset="Vistra Corp. Common Stock (VST) [ST]", date="01/16/2026"):
    return {
        "asset": asset,
        "type": "Purchase",
        "date": date,
        "notif_date": "01/23/2026",
        "amount_letter": "C",
    }


def test_quoted_nickname_metadata_matches_filed_name():
    valid, rejections = validate_transactions(
        "9116212",
        "Charles J Fleischmann",
        [_tx()],
        "07/08/2026",
        'Charles J. "Chuck" Fleischmann',
    )
    assert len(valid) == 1
    assert rejections == {}


def test_genuine_member_mismatch_still_rejected():
    valid, rejections = validate_transactions(
        "9116212", "Nancy Pelosi", [_tx()], "07/08/2026", "Charles J. Fleischmann"
    )
    assert valid == []
    assert rejections.get("member_mismatch") == 1


def test_impossible_date_rejects_only_that_row():
    valid, rejections = validate_transactions(
        "20033889",
        "Steve Cohen",
        [_tx(), _tx(date="12/26/2026")],
        "02/09/2026",
        "Steve Cohen",
    )
    assert len(valid) == 1
    assert rejections.get("date_out_of_window") == 1


def test_hal_harold_nickname_matches():
    valid, rejections = validate_transactions(
        "9115808",
        "Rep. Hal Rogers",
        [_tx()],
        "07/08/2026",
        "Harold Dallas Rogers",
    )
    assert len(valid) == 1
    assert rejections == {}


def test_surname_only_filing_matches_metadata():
    valid, rejections = validate_transactions(
        "9116217", "Malliotakis", [_tx()], "07/08/2026", "Nicole Malliotakis"
    )
    assert len(valid) == 1
    assert rejections == {}


def test_surname_only_mismatch_still_rejected():
    valid, rejections = validate_transactions(
        "9116217", "Smith", [_tx()], "07/08/2026", "Nicole Malliotakis"
    )
    assert valid == []
    assert rejections.get("member_mismatch") == 1


def test_leading_title_stripped_from_member():
    valid, rejections = validate_transactions(
        "20033725",
        "Hon. Nancy Pelosi",
        [
            {
                "asset": "Vistra Corp. Common Stock",
                "type": "Purchase",
                "date": "01/16/2026",
                "notif_date": "01/23/2026",
                "amount_letter": "C",
            }
        ],
        "01/23/2026",
        "Nancy Pelosi",
    )
    assert len(valid) == 1
    assert valid[0]["member"] == "Nancy Pelosi"
    assert rejections == {}


def test_variant_spelling_persists_metadata_form():
    valid, rejections = validate_transactions(
        "9116211",
        "Mike McCaul",
        [
            {
                "asset": "Fidelity Fund",
                "type": "Purchase",
                "date": "06/01/2026",
                "notif_date": "06/20/2026",
                "amount_letter": "C",
            }
        ],
        "07/08/2026",
        "Michael T. McCaul",
    )
    assert len(valid) == 1
    assert valid[0]["member"] == "Michael T. McCaul"
    assert rejections == {}
