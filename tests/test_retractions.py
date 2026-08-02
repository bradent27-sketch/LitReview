from litdesk import retractions


def test_is_retraction_flagged_detects_retraction():
    raw = {"pubTypeList": {"pubType": ["Journal Article", "Retracted Publication"]}}
    flagged, note = retractions.is_retraction_flagged(raw)
    assert flagged is True
    assert note == "Retracted Publication"


def test_is_retraction_flagged_detects_correction_and_expression_of_concern():
    assert retractions.is_retraction_flagged({"pubTypeList": {"pubType": ["Published Erratum"]}})[0] is False
    assert retractions.is_retraction_flagged({"pubTypeList": {"pubType": ["Correction"]}})[0] is True
    assert retractions.is_retraction_flagged({"pubTypeList": {"pubType": ["Expression of Concern"]}})[0] is True


def test_is_retraction_flagged_false_for_normal_article():
    raw = {"pubTypeList": {"pubType": ["Journal Article"]}}
    assert retractions.is_retraction_flagged(raw) == (False, None)


def test_is_retraction_flagged_handles_missing_or_malformed_raw():
    assert retractions.is_retraction_flagged({}) == (False, None)
    assert retractions.is_retraction_flagged(None) == (False, None)


def test_check_all_papers_flags_and_notifies(db_conn, insert_paper):
    import json

    pid = insert_paper(
        db_conn, title="Retracted Paper",
        raw=json.dumps({"pubTypeList": {"pubType": ["Retracted Publication"]}}),
    )
    clean_id = insert_paper(
        db_conn, title="Fine Paper",
        raw=json.dumps({"pubTypeList": {"pubType": ["Journal Article"]}}),
    )

    flagged = retractions.check_all_papers(db_conn)

    assert len(flagged) == 1
    assert flagged[0]["id"] == pid
    assert flagged[0]["note"] == "Retracted Publication"

    row = db_conn.execute("SELECT retracted, retraction_note FROM papers WHERE id = ?", (pid,)).fetchone()
    assert row["retracted"] == 1
    assert row["retraction_note"] == "Retracted Publication"

    clean_row = db_conn.execute("SELECT retracted FROM papers WHERE id = ?", (clean_id,)).fetchone()
    assert clean_row["retracted"] == 0

    notif = db_conn.execute("SELECT * FROM notifications WHERE paper_id = ?", (pid,)).fetchone()
    assert notif["type"] == "retraction"


def test_check_all_papers_does_not_reflag_already_flagged(db_conn, insert_paper):
    import json

    pid = insert_paper(
        db_conn, title="Already Flagged", retracted=1, retraction_note="Retracted Publication",
        raw=json.dumps({"pubTypeList": {"pubType": ["Retracted Publication"]}}),
    )
    flagged = retractions.check_all_papers(db_conn)
    assert flagged == []
    notif_count = db_conn.execute("SELECT COUNT(*) AS n FROM notifications WHERE paper_id = ?", (pid,)).fetchone()["n"]
    assert notif_count == 0
