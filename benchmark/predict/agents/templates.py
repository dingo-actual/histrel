#!/usr/bin/env python3
"""
Message template for benchmark agents.

Provides the benchmark message template, plus a rendering helper.

    Important Variables:
        MESSAGE_TEMPLATE (str): Template specifying SWEBench problem.

    Important Functions:
        render_template(template: str, **kwargs: str) -> str: Render template from inputs.
"""

from __future__ import annotations

MESSAGE_TEMPLATE = """## Repo

{repo}

## Problem

{problem_statement}

## Hints

{hints_text}
"""


def render_template(template: str, **kwargs: str) -> str:
    """Render a template string with provided keyword arguments.

    Args:
        template (str): Template string.
        **kwargs (str): Values to substitute into placeholders.

    Returns:
        (str) Rendered template string with all available placeholders substituted.

    Example Usage:
        ```python
            >>> render_template("Hello {{ name }}!", name="World")
            'Hello World!'
            >>> render_template("Missing {{ foo }} and {{ bar }}", foo="X", bar="")
            'Missing X and '
        ```
    """
    safe_kwargs = {k: (v if v else "") for k, v in kwargs.items()}
    try:
        return template.format(**safe_kwargs)
    except KeyError:
        return template.format_map(safe_kwargs)
