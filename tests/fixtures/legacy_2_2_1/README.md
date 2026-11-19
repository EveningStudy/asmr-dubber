# ASMR Dubber 2.2.1 projects

These three projects were written by the released 2.2.1 code, not by hand, so they
pin the exact manifest format (`schema_version` 3) and the `tts_cache_key` values that
2.2.1 computed. `tests/test_legacy_2_2_1_projects.py` opens them with the current code.

To regenerate (only ever with the 2.2.1 code):

```bash
git archive v2.2.1 src/asmr_dubber | tar -x -C /tmp/v221
python tests/fixtures/legacy_2_2_1/make_fixture.py /tmp/v221/src tests/fixtures/legacy_2_2_1
```

The audio is synthetic tones; no real recording is included.
