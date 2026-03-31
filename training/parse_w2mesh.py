"""
parse_w2mesh.py
Extract vertex/face data from Witcher 3 .w2mesh (CR2W v163) files
and export as GLB for viewing.

W3 mesh CR2W structure:
  - Header + string/name tables (same as .w2l)
  - Chunks: CMesh object with embedded vertex/index buffers
  - Buffer section: raw vertex + index data

Usage:
    python parse_w2mesh.py <file.w2mesh> [--out file.glb]
    python parse_w2mesh.py <directory> --out combined.glb
"""

import argparse
import struct
import sys
import numpy as np
from pathlib import Path

CR2W_MAGIC = b"CR2W"


def _read_header(data):
    """Parse CR2W header, return table descriptors."""
    if data[:4] != CR2W_MAGIC:
        return None
    version = struct.unpack_from("<I", data, 4)[0]
    if version != 163:
        return None

    tables = {}
    names = ["strings", "names", "resources", "objects", "chunks", "buffers"]
    for i, name in enumerate(names):
        off = 0x28 + i * 12
        t_off, t_cnt, t_crc = struct.unpack_from("<III", data, off)
        if t_off or t_cnt:
            tables[name] = {"offset": t_off, "count": t_cnt}
    return tables


def _read_names(data, tables):
    """Extract name strings."""
    str_t = tables.get("strings", {})
    nam_t = tables.get("names", {})
    if not str_t or not nam_t:
        return []

    str_off = str_t["offset"]
    nam_off = nam_t["offset"]
    blob = data[str_off:nam_off]

    strtable = {}
    i = 0
    while i < len(blob):
        end = blob.find(b"\x00", i)
        if end < 0:
            end = len(blob)
        strtable[i] = blob[i:end].decode("utf-8", errors="replace")
        i = end + 1

    names = []
    for n in range(nam_t["count"]):
        s_off, _ = struct.unpack_from("<II", data, nam_off + n * 8)
        names.append(strtable.get(s_off, ""))
    return names


def _find_buffers(data, tables):
    """
    Find raw data buffers in the CR2W file.
    Buffer table entries: 24 bytes each (offset, diskSize, memSize, crc, ...)
    The actual buffer data follows the main CR2W data section.
    """
    buf_t = tables.get("buffers")
    if not buf_t or buf_t["count"] == 0:
        return []

    buffers = []
    for i in range(buf_t["count"]):
        entry_off = buf_t["offset"] + i * 24
        if entry_off + 24 > len(data):
            break
        vals = struct.unpack_from("<IIIIII", data, entry_off)
        # Try different interpretations of the 24-byte entry
        # Common: offset(4), diskSize(4), memSize(4), crc(4), ...
        buf_offset = vals[0]
        disk_size = vals[1]
        mem_size = vals[2]
        buffers.append({
            "offset": buf_offset,
            "disk_size": disk_size,
            "mem_size": mem_size,
        })
    return buffers


def _extract_mesh_data(data):
    """
    Try to extract vertex positions and triangle indices from raw CR2W mesh data.
    Strategy: scan for the vertex buffer (dense float32 triplets in a plausible range)
    and the index buffer (dense uint16/uint32 values).
    """
    tables = _read_header(data)
    if tables is None:
        return None, None

    names = _read_names(data, tables)

    # Find vertex count and index count from chunk properties
    # Search for known property names
    vert_count = 0
    idx_count = 0

    # Look for "numVertices" and "numIndices" patterns
    for pattern, attr in [(b"numVertices", "vert"), (b"numIndices", "idx")]:
        pos = data.find(pattern)
        if pos > 0:
            # Value is typically a uint32 shortly after the name
            # Scan forward for a reasonable count
            for off in range(pos + len(pattern), min(pos + len(pattern) + 32, len(data) - 4)):
                val = struct.unpack_from("<I", data, off)[0]
                if 3 < val < 500000:
                    if attr == "vert":
                        vert_count = val
                    else:
                        idx_count = val
                    break

    # Scan for vertex data: look for dense blocks of float32 in range [-100, 100]
    # W3 meshes use quantized vertex formats, but let's try float32 first
    file_size = len(data)
    best_verts = None
    best_score = 0

    # Try to find vertex buffer by scanning for float32 XYZ triplets
    # Heuristic: find a contiguous region where every 12 bytes is 3 valid floats
    SCAN_START = len(data) // 2  # vertices are usually in the second half
    STRIDE_CANDIDATES = [12, 24, 28, 32, 36, 40, 44, 48, 52, 56]

    for stride in STRIDE_CANDIDATES:
        for start in range(SCAN_START, file_size - stride * 10, 4):
            # Quick check first 3 vertices
            valid = True
            for vi in range(3):
                off = start + vi * stride
                if off + 12 > file_size:
                    valid = False
                    break
                x, y, z = struct.unpack_from("<fff", data, off)
                if not (-500 < x < 500 and -500 < y < 500 and -500 < z < 500):
                    valid = False
                    break
                if x == 0 and y == 0 and z == 0:
                    valid = False
                    break
            if not valid:
                continue

            # Count how many consecutive valid vertices
            count = 0
            for vi in range(min(200000, (file_size - start) // stride)):
                off = start + vi * stride
                if off + 12 > file_size:
                    break
                x, y, z = struct.unpack_from("<fff", data, off)
                if -500 < x < 500 and -500 < y < 500 and -500 < z < 500:
                    count += 1
                else:
                    break

            if count > best_score and count >= 10:
                best_score = count
                verts = []
                for vi in range(count):
                    off = start + vi * stride
                    x, y, z = struct.unpack_from("<fff", data, off)
                    verts.append([x, y, z])
                best_verts = np.array(verts, dtype=np.float32)

            if best_score > 100:
                break
        if best_score > 100:
            break

    if best_verts is None or len(best_verts) < 3:
        return None, None

    # Find index buffer: look for uint16 triples that reference valid vertex indices
    n_verts = len(best_verts)
    best_faces = None
    best_face_count = 0

    for start in range(SCAN_START, file_size - 6, 2):
        # Check first triangle
        i0, i1, i2 = struct.unpack_from("<HHH", data, start)
        if i0 >= n_verts or i1 >= n_verts or i2 >= n_verts:
            continue
        if i0 == i1 or i1 == i2 or i0 == i2:
            continue

        # Count valid consecutive triangles
        faces = []
        off = start
        while off + 6 <= file_size:
            i0, i1, i2 = struct.unpack_from("<HHH", data, off)
            if i0 >= n_verts or i1 >= n_verts or i2 >= n_verts:
                break
            faces.append([i0, i1, i2])
            off += 6

        if len(faces) > best_face_count:
            best_face_count = len(faces)
            best_faces = np.array(faces, dtype=np.int64)

        if best_face_count > 100:
            break

    return best_verts, best_faces


def w2mesh_to_glb(mesh_path: Path, output_path: Path) -> dict:
    """Convert a .w2mesh file to GLB."""
    import trimesh

    data = mesh_path.read_bytes()
    verts, faces = _extract_mesh_data(data)

    if verts is None or faces is None:
        print(f"  [FAIL] Could not extract mesh from {mesh_path.name}")
        return None

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    trimesh.repair.fix_normals(mesh)

    # Color by height (Y) for visualization
    y = mesh.vertices[:, 1]
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
    colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    for fi in range(len(mesh.faces)):
        avg_y = y_norm[mesh.faces[fi]].mean()
        r = int(180 - avg_y * 60)
        g = int(160 - avg_y * 40)
        b = int(140 - avg_y * 30)
        colors[fi] = [r, g, b, 255]
    mesh.visual.face_colors = colors

    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name=mesh_path.stem)
    scene.export(str(output_path))

    return {
        "name": mesh_path.stem,
        "vertices": len(verts),
        "faces": len(faces),
        "bounds": {
            "x": [float(verts[:, 0].min()), float(verts[:, 0].max())],
            "y": [float(verts[:, 1].min()), float(verts[:, 1].max())],
            "z": [float(verts[:, 2].min()), float(verts[:, 2].max())],
        },
        "output": str(output_path),
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description="W3 .w2mesh -> GLB converter")
    ap.add_argument("input", help=".w2mesh file or directory")
    ap.add_argument("--out", default=None, help="Output GLB path")
    args = ap.parse_args()

    inp = Path(args.input)
    if inp.is_file():
        out = Path(args.out) if args.out else inp.with_suffix(".glb")
        print(f"Converting: {inp.name}")
        result = w2mesh_to_glb(inp, out)
        if result:
            print(f"  Vertices: {result['vertices']:,}")
            print(f"  Faces: {result['faces']:,}")
            print(f"  Output: {out}")
    elif inp.is_dir():
        files = sorted(inp.glob("*.w2mesh"))[:10]
        print(f"Converting {len(files)} files from {inp}")
        import trimesh
        combined = trimesh.Scene()
        for f in files:
            print(f"\n  {f.name}...")
            result = w2mesh_to_glb(f, f.with_suffix(".glb"))
            if result:
                print(f"    {result['vertices']:,} verts, {result['faces']:,} faces")


if __name__ == "__main__":
    main()
