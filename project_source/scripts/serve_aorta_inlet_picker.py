"""Loopback-only E-029 geometry picker; explicit local selection saves only."""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import uuid

import numpy as np

from scripts.build_aorta_inlet_picker import make_selection, sha


def validate_payload(payload):
    if not isinstance(payload, dict) or set(payload) - {'sample_index', 'note'} or 'sample_index' not in payload:
        raise ValueError('Expected sample_index and optional note only')
    if type(payload['sample_index']) is not int:
        raise ValueError('sample_index must be an integer')
    note = payload.get('note', '')
    if not isinstance(note, str) or len(note) > 500:
        raise ValueError('Note must be text of at most 500 characters')
    return payload['sample_index'], note


def build_server(root, port, *, test_mode=False):
    root = Path(root).resolve()
    manifest = json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
    for name in ('index.html', 'geometry.json', 'source_samples.npy', 'original_coordinate_rows.npy'):
        if sha(root/name) != manifest['files'][name]['sha256']:
            raise ValueError('Picker artifact changed: '+name)
    metadata = json.loads((root/'geometry.json').read_text(encoding='utf-8'))
    points = np.load(root/'source_samples.npy', allow_pickle=False)
    ids = np.load(root/'original_coordinate_rows.npy', allow_pickle=False)
    html = (root/'index.html').read_text(encoding='utf-8')
    destination = root/('validation/browser_test_selections' if test_mode else 'user_selections')
    destination.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def allowed_host(self):
            return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

        def reply(self, code, data, content_type='application/json; charset=utf-8'):
            body = data.encode('utf-8') if isinstance(data, str) else json.dumps(data, allow_nan=False).encode('utf-8')
            self.send_response(code); self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body))); self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff'); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if not self.allowed_host():
                self.reply(403, {'error': 'Loopback host only'}); return
            if self.path in ('/', '/index.html'):
                runtime = json.dumps(dict(csrf=token, test_mode=test_mode))
                self.reply(200, html.replace('const LOCAL_RUNTIME = null; // SERVER_RUNTIME',
                    'const LOCAL_RUNTIME = '+runtime+'; // SERVER_RUNTIME'), 'text/html; charset=utf-8')
            elif self.path == '/health':
                self.reply(200, dict(status='ready', test_mode=test_mode, point_count=len(points)))
            elif self.path == '/selection':
                latest = destination/'latest_selection.json'
                self.reply(200, dict(selection=json.loads(latest.read_text(encoding='utf-8')) if latest.exists() else None))
            elif self.path == '/favicon.ico':
                self.reply(204, '')
            else:
                self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            valid_origins = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
            if not self.allowed_host() or self.headers.get('Origin') not in valid_origins:
                self.reply(403, {'error': 'Same-origin local save required'}); return
            if self.path != '/save-selection':
                self.reply(404, {'error': 'Not found'}); return
            if not hmac.compare_digest(self.headers.get('X-MRSim-Token', ''), token):
                self.reply(403, {'error': 'Invalid local save token'}); return
            if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
                self.reply(415, {'error': 'JSON required'}); return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4096:
                    self.reply(413, {'error': 'Request size must be 1–4096 bytes'}); return
                index, note = validate_payload(json.loads(self.rfile.read(size)))
                selected = make_selection(index, points, ids, metadata, test_mode=test_mode, note=note)
            except (ValueError, UnicodeDecodeError, TypeError) as exc:
                self.reply(400, {'error': str(exc)}); return
            name = 'selection_'+selected['created_utc'].replace(':', '').replace('.', '_')+'_'+uuid.uuid4().hex[:8]+'.json'
            raw = json.dumps(selected, indent=2, allow_nan=False)
            with (destination/name).open('x', encoding='utf-8') as stream:
                stream.write(raw)
            temp = destination/('latest_'+uuid.uuid4().hex+'.tmp')
            with temp.open('x', encoding='utf-8') as stream:
                stream.write(raw)
            temp.replace(destination/'latest_selection.json')
            self.reply(201, dict(filename=name, selection=selected))

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=4175)
    parser.add_argument('--test-mode', action='store_true')
    args = parser.parse_args()
    server = build_server(args.root, args.port, test_mode=args.test_mode)
    print(f'Aortic inlet picker: http://127.0.0.1:{server.server_port}/ (test_mode={args.test_mode})', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
