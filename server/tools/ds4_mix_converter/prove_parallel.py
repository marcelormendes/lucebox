#!/usr/bin/env python3
"""Soulf-only bounded real-source byte/time proof; never launches a full conversion."""
import argparse
import filecmp
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ('old', 'new', 'source', 'imatrix', 'evidence'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=False)
    evidence = args.evidence
    manifest = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'old_binary_sha256': digest(args.old), 'new_binary_sha256': digest(args.new),
                'imatrix_sha256': digest(args.imatrix), 'source_files': {}, 'lanes': []}
    for name in ('config.json', 'tokenizer.json', 'model.safetensors.index.json'):
        manifest['source_files'][name] = digest(args.source/name)
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    for name, binary, extra in [('old', args.old, []), ('default', args.new, []),
                               ('one', args.new, ['--encode-threads', '1']),
                               ('eight', args.new, ['--encode-threads', '8']),
                               ('eight-repeat', args.new, ['--encode-threads', '8'])]:
        output = evidence/(name+'.gguf')
        command = [str(binary), '--input', str(args.source), '--output', str(output),
                   '--imatrix', str(args.imatrix), '--layer-count', '1', '--expert-limit', '17',
                   '--experts-only'] + extra
        lane = {'name': name, 'command': command, 'samples': []}
        print('START', name, flush=True)
        started = time.monotonic()
        proc = subprocess.Popen(['/usr/bin/time', '-v', '-o', str(evidence/(name+'.time'))] + command,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        (evidence/(name+'.time-pid')).write_text(str(proc.pid)+'\n')
        def sample():
            while proc.poll() is None:
                try:
                    children = Path(f'/proc/{proc.pid}/task/{proc.pid}/children').read_text().split()
                    for child in children:
                        tasks = list(Path(f'/proc/{child}/task').iterdir())
                        running = 0
                        for task in tasks:
                            try:
                                state = (task/'stat').read_text().split(') ', 1)[1].split()[0]
                                running += state == 'R'
                            except FileNotFoundError:
                                pass
                        lane['samples'].append([round(time.monotonic()-started, 3), len(tasks), running])
                except FileNotFoundError:
                    pass
                time.sleep(.2)
        watcher = threading.Thread(target=sample)
        watcher.start()
        with open(evidence/(name+'.log'), 'w') as log:
            for line in proc.stdout:
                log.write(f'{time.monotonic()-started:.6f} {line}')
        lane['exit'] = proc.wait(); watcher.join()
        lane['elapsed_seconds'] = time.monotonic()-started
        if lane['exit'] == 0:
            lane['gguf_sha256'] = digest(output)
            lane['gumix_sha256'] = digest(str(output)+'.gumix.bin')
            lane['gguf_bytes'] = output.stat().st_size
            if name != 'old':
                lane['gguf_equal_old'] = filecmp.cmp(evidence/'old.gguf', output, shallow=False)
                lane['gumix_equal_old'] = filecmp.cmp(str(evidence/'old.gguf')+'.gumix.bin',
                                                    str(output)+'.gumix.bin', shallow=False)
        manifest['lanes'].append(lane)
        (evidence/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print('FINISH', name, 'exit', lane['exit'], 'elapsed', round(lane['elapsed_seconds'], 3), flush=True)
        if lane['exit'] or lane.get('gguf_equal_old') is False or lane.get('gumix_equal_old') is False:
            raise SystemExit('failed lane; no next lane launched')
    if digest(args.old) != manifest['old_binary_sha256']:
        raise SystemExit('old binary changed externally during proof')
    print('PASS: all five complete GGUF and GUMIX artifacts are byte-identical', flush=True)


if __name__ == '__main__':
    main()
