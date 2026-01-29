MOCK_RESPONSES = {
    "gpt-4o": {
        "simple": "The answer to 2+2 is 4.",
        "code": "```python\ndef sort_list(lst):\n    return sorted(lst)\n```",
        "reasoning": "TCP provides reliable, ordered delivery with connection establishment. UDP is connectionless and faster but unreliable.",
    },
    "claude-sonnet": {
        "simple": "2+2 equals 4.",
        "code": "```python\ndef sort_list(items):\n    return sorted(items)\n```",
        "reasoning": "TCP is reliable with guaranteed delivery. UDP is fast but packets may be lost.",
    },
    "gpt-4o-mini": {
        "simple": "4",
        "code": "def sort(l): return sorted(l)",
        "reasoning": "TCP = reliable, UDP = fast.",
    },
}

ERROR_RESPONSES = {
    "rate_limit": {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
    "timeout": {"error": {"message": "Request timeout", "type": "timeout_error"}},
    "invalid_key": {"error": {"message": "Invalid API key", "type": "authentication_error"}},
}
