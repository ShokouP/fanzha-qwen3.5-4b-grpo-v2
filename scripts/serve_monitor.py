"""Read-only localhost GRPO monitor; point --out to a training output directory."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True, type=Path)
parser.add_argument('--port', type=int, default=8787)
args = parser.parse_args()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ('/', '/grpo_v2', '/grpo_v2/'):
            data = (Path(__file__).resolve().parents[1] / 'monitor' / 'index.html').read_bytes()
            content = 'text/html; charset=utf-8'
        elif path == '/grpo_v2/status.json':
            try:
                summary = json.loads((args.out / 'summary.json').read_text(encoding='utf-8-sig'))
            except (OSError, ValueError):
                summary = {}
            try:
                train = [json.loads(line) for line in (args.out / 'metrics.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
            except (OSError, ValueError):
                train = []
            status = summary.get('status', 'waiting')
            obj = {'stage': status, 'tone': 'done' if status == 'completed' else 'live',
                   'step': train[-1]['step'] if train else 0,
                   'target': summary.get('config', {}).get('steps', 40),
                   'updated_at': None, 'best_step': summary.get('best_step'),
                   'status': status, 'train': train, 'validation': summary.get('evals', [])}
            data = json.dumps(obj, ensure_ascii=False).encode('utf-8')
            content = 'application/json; charset=utf-8'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', content)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

if __name__ == '__main__':
    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        print(f'http://127.0.0.1:{args.port}/grpo_v2/', flush=True)
        server.serve_forever()
