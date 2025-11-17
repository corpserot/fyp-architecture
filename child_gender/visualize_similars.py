from http.server import SimpleHTTPRequestHandler, HTTPServer
import os

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
FILE_TO_SERVE_ON_ROOT = "visualize_similars.html"

class CustomHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SCRIPTDIR, **kwargs)

    def do_GET(self):
        if self.path == '/':
            self.path = '/' + FILE_TO_SERVE_ON_ROOT
        return SimpleHTTPRequestHandler.do_GET(self)

def run(server_class=HTTPServer, handler_class=CustomHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Serving files from {SCRIPTDIR} on http://localhost:{port}/')
    print(f'Access {FILE_TO_SERVE_ON_ROOT} at http://localhost:{port}/')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
