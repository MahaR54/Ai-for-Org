import webview
import os

# Serve local files via built-in HTTP server so relative assets work
here = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    window = webview.create_window(
        title="Reflect",
        url=os.path.join(here, "index.html"),
        width=1280,
        height=800,
        resizable=True
    )
    # http_server=True lets index.html load ./assets/*
    webview.start(http_server=True, gui="edgechromium")  # falls back automatically if not available
