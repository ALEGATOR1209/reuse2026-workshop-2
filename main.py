"""MNIST on a fruit fly brain: proof of concept.

    uv run main.py download                     # connectome + annotations + MNIST (~140 MB)
    uv run main.py simulate --train 2000 --test 500
    uv run main.py train                        # fit readouts on the last simulation
    uv run main.py all --train 2000 --test 500  # all three steps
    uv run main.py serve                        # web app -> http://127.0.0.1:8000 and http://<LAN IP>:8000
    uv run main.py poses                        # fly poses for the browser (flymnist/static/fly-poses.json)
    uv run main.py site                         # static student version (no server, e.g. GitHub Pages) -> site/
    uv run main.py pretrain                     # pre-trained gesture model from HaGRIDv2 photos (flymnist/pretrain.py)
    uv run main.py skeleton-test                # try the stick-figure gesture model -> http://127.0.0.1:8002

    # split brain: cut all left<->right connections, then train left-only / right-only readouts
    uv run main.py simulate --split --train 3000 --test 1000
    uv run main.py hemispheres --split          # or without --split for the intact brain
"""

import argparse
from pathlib import Path

from flymnist import data, readout, simulate
from flymnist.retina import INPUT_LAYERS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["download", "simulate", "train", "all", "hemispheres", "serve", "poses", "site", "pretrain", "skeleton-test"])
    ap.add_argument("--train", type=int, default=2000, help="training digits to simulate")
    ap.add_argument("--test", type=int, default=500, help="test digits to simulate")
    ap.add_argument("--layer", choices=list(INPUT_LAYERS), default="lamina", help="where the image enters the brain")
    ap.add_argument("--rate", type=float, default=500.0, help="firing rate (Hz) of an input neuron on a white pixel")
    ap.add_argument("--ms", type=float, default=100.0, help="how long each digit is shown (simulated ms)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--shuffle", action="store_true", help="control: randomly rewire the connectome")
    ap.add_argument("--split", action="store_true", help="split brain: cut all left<->right connections")
    ap.add_argument("--port", type=int, default=8000, help="web app port")
    ap.add_argument("--host", default="0.0.0.0",
                    help="web app address; 0.0.0.0 = reachable from the local network, 127.0.0.1 = this computer only")
    ap.add_argument("--https-port", type=int, default=8443,
                    help="also serve HTTPS here (self-signed), needed for webcams on other devices; 0 = off")
    ap.add_argument("--tls-dir", default=None, help="where to keep the HTTPS certificate (default: data/tls)")
    ap.add_argument("--out", default="site", help="site: output folder")
    ap.add_argument("--url", default=None, help="site: the published address, for the QR code on the slides")
    ap.add_argument("--step", default="all", choices=["all", "download", "pose", "build", "simulate", "train", "skeleton"],
                    help="pretrain: run one step only")
    a = ap.parse_args()

    if a.command == "serve":
        serve(a)
        return
    if a.command == "poses":   # fly poses for animating the body without physics (flymnist/static/flypose.js)
        from flymnist import poses
        poses.export()
        return
    if a.command == "site":    # static student version: no server needed (flymnist/site.py)
        from flymnist import site
        site.build(a.out, a.layer, a.rate, a.ms, a.shuffle, url=a.url)
        return

    if a.command == "skeleton-test":   # stick-figure gesture test page, this computer only (flymnist/skeleton_server.py)
        from flymnist import skeleton_server
        skeleton_server.serve(8002 if a.port == 8000 else a.port)
        return
    if a.command == "pretrain":   # alternative class model from HaGRIDv2 photos; never published (flymnist/pretrain.py)
        from flymnist import pretrain
        if a.step == "skeleton":      # stick-figure model (flymnist/pretrain_skeleton.py), needs download + pose first
            from flymnist import pretrain_skeleton
            pretrain_skeleton.run()
        else:
            pretrain.run(a.step)
        return

    path = simulate.feature_path(a.layer, a.rate, a.ms, a.shuffle, a.split)
    if a.command in ("download", "all"):
        data.download()
    if a.command in ("simulate", "all"):
        simulate.run(a.train, a.test, a.layer, a.rate, a.ms, a.batch, a.shuffle, split=a.split)
    if a.command in ("train", "all"):
        readout.run(path, out_json=path.with_suffix(".results.json"))
    if a.command == "hemispheres":
        readout.run_hemispheres(path)


def serve(a):
    import asyncio

    import uvicorn
    from flymnist.server import create_app, lan_ip
    from flymnist.tls import ensure_cert

    ip = lan_ip() if a.host == "0.0.0.0" else None
    if a.host == "0.0.0.0" and not ip:
        print("  no local network found: serving on this computer only (no HTTPS)", flush=True)
    https_port = a.https_port if ip else 0
    if https_port:
        try:
            cert, key = ensure_cert(ip, Path(a.tls_dir)) if a.tls_dir else ensure_cert(ip)
        except Exception as e:  # noqa: BLE001  (no certificate: still serve plain HTTP)
            print(f"  HTTPS disabled: {e}; phones can't use their camera, everything else works", flush=True)
            https_port = 0
    app = create_app(a.layer, a.rate, a.ms, a.shuffle, host=a.host, port=a.port, https_port=https_port)
    configs = [uvicorn.Config(app, host=a.host, port=a.port)]
    if https_port:
        configs.append(uvicorn.Config(app, host=a.host, port=https_port, ssl_certfile=str(cert), ssl_keyfile=str(key)))

    print(f"\n  on this computer:      http://127.0.0.1:{a.port}", flush=True)
    if ip:
        print(f"  on the local network:  http://{ip}:{a.port}", flush=True)
    if https_port:
        print(f"  with camera (https):   https://{ip}:{https_port}   (QR code in the page header;"
              " browsers warn about the self-signed certificate once)\n", flush=True)

    servers = [uvicorn.Server(c) for c in configs]

    async def run_all():  # http and https in one process, sharing one brain; stop both together
        tasks = [asyncio.create_task(s.serve()) for s in servers]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for s in servers:
            s.should_exit = True
        await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
