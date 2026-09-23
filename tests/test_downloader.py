import asyncio
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from amlc.data.downloader import MANIFEST, download_all, load_done
from amlc.data.io import image_index


def _jpeg(size=(800, 600)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, format="JPEG")
    return buf.getvalue()


JPEG = _jpeg()
hits: dict[str, int] = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        hits[self.path] = hits.get(self.path, 0) + 1
        if self.path.startswith("/flaky") and hits[self.path] < 3:
            self.send_response(503)
            self.send_header("Retry-After", "0")
            self.end_headers()
            return
        if self.path.startswith("/missing"):
            self.send_response(404)
            self.end_headers()
            return
        if self.path.startswith("/corrupt"):
            body = b"not an image"
        else:
            body = JPEG
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_download_retry_resume_and_failures(server, tmp_path):
    hits.clear()
    urls = [f"{server}/img/{i}.jpg" for i in range(50)]
    urls += [urls[0], f"{server}/flaky/a.jpg", f"{server}/missing/b.jpg", f"{server}/corrupt/c.jpg", None]

    stats = asyncio.run(download_all(urls, tmp_path, concurrency=16, retries=3, backoff=0.01, max_side=256))
    assert stats.ok == 51  # 50 + flaky after retries; duplicate url fetched once
    assert stats.failed == 2  # 404 + undecodable
    assert hits["/flaky/a.jpg"] == 3
    assert hits["/img/0.jpg"] == 1

    with Image.open(tmp_path / "0.jpg") as im:
        assert max(im.size) == 256  # resized
    recs = [json.loads(line) for line in (tmp_path / MANIFEST).read_text().splitlines()]
    assert all(len(r["sha256"]) == 64 for r in recs)
    assert len((tmp_path / "failed.csv").read_text().splitlines()) == 2
    assert not list(tmp_path.glob("*.part"))

    # resume: second run fetches only the previously failed urls
    before = dict(hits)
    stats2 = asyncio.run(download_all(urls, tmp_path, concurrency=16, retries=0, backoff=0.01))
    assert stats2.skipped == 51 and stats2.ok == 0
    assert hits["/img/5.jpg"] == before["/img/5.jpg"]
    assert len(load_done(tmp_path)) == 51

    idx = image_index(tmp_path)
    assert idx.height == 51 and idx["image_path"].str.ends_with(".jpg").all()


def test_same_basename_different_paths_do_not_collide(server, tmp_path):
    urls = [f"{server}/x/same.jpg", f"{server}/y/same.jpg"]
    asyncio.run(download_all(urls, tmp_path, concurrency=2, backoff=0.01))
    assert len(list(tmp_path.glob("*same.jpg"))) == 2
