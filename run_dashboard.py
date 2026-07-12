import http.server
import socketserver
import os
import webbrowser
import sys

PORT = 8000
DIRECTORY = "web"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def main():
    # Ensure working directory is the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    if not os.path.exists(DIRECTORY):
        print(f"Error: Directory '{DIRECTORY}' not found. Make sure you are running from the project root.", file=sys.stderr)
        sys.exit(1)
        
    # Find an open port
    port = PORT
    server = None
    while port < 8100:
        try:
            server = socketserver.TCPServer(("", port), Handler)
            break
        except OSError:
            port += 1
            
    if server is None:
        print("Error: Could not find an open port between 8000 and 8100.", file=sys.stderr)
        sys.exit(1)
        
    url = f"http://localhost:{port}/"
    print(f"\n==========================================")
    print(f"   GNN XAI Dashboard Server Started")
    print(f"==========================================")
    print(f"Dashboard URL: {url}")
    print(f"Press Ctrl+C to terminate the server.\n")
    
    # Automatically open standard browser
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server.")
        server.server_close()

if __name__ == "__main__":
    main()
