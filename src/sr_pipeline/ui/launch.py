"""``sr-ui``: start the Streamlit run viewer (headless: no first-run email prompt, no
telemetry) and open it in the browser."""

import sys
import threading
import webbrowser
from pathlib import Path

PORT = "8501"


def main() -> None:
    from streamlit.web import cli as stcli

    app = Path(__file__).with_name("app.py")
    args = sys.argv[1:]
    port = args[args.index("--server.port") + 1] if "--server.port" in args else PORT
    if "--no-browser" in args:
        args.remove("--no-browser")
    else:
        threading.Timer(2.5, webbrowser.open, [f"http://localhost:{port}"]).start()
    sys.argv = ["streamlit", "run", str(app),
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false",
                "--theme.base", "light",
                "--theme.primaryColor", "#2a78d6",
                "--client.toolbarMode", "minimal",
                *args]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
