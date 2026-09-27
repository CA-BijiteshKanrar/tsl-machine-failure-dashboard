"""Small client for the project's Azure ML managed online scoring endpoint."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def score_records(records, scoring_uri: str, endpoint_key: str) -> dict:
    if not records:
        raise ValueError("No rows to score")
    if not scoring_uri.startswith("https://"):
        raise ValueError("Azure scoring URI must use HTTPS")
    request = Request(
        scoring_uri,
        data=json.dumps({"data": records}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {endpoint_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"Azure scoring returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Azure scoring could not be reached: {exc.reason}") from exc
    if isinstance(result, str):
        result = json.loads(result)
    if not isinstance(result, dict) or "failure_probability" not in result:
        raise RuntimeError("Unexpected Azure scoring response format")
    if len(result["failure_probability"]) != len(records):
        raise RuntimeError("Azure response row count does not match the request")
    return result
