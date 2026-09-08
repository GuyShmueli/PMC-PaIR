from __future__ import annotations

import argparse


class SingleUseOption(argparse.Action):
    """Reject repeated use of an option that denotes one canonical input."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string or self.dest} may be specified only once")
        setattr(namespace, self.dest, values)


__all__ = ["SingleUseOption"]
