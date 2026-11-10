"""``sr-ui``: start the Streamlit run viewer (no telemetry, light theme)."""

import sys
from pathlib import Path


def main() -> None:
    from streamlit.web import cli as stcli

    app = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app),
                "--browser.gatherUsageStats", "false",
                "--theme.base", "light",
                "--theme.primaryColor", "#2a78d6",
                *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
