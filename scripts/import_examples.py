"""Import the organizer's sample ZIP, omitting macOS metadata."""
import argparse
import zipfile
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--destination", type=Path, default=Path("examples/tasks"))
    args = parser.parse_args()
    root = args.destination.resolve()
    count = 0
    with zipfile.ZipFile(args.archive) as archive:
        for entry in archive.infolist():
            parts = PurePosixPath(entry.filename).parts
            if entry.is_dir() or len(parts) < 3 or parts[0] != "task-dir":
                continue
            if any(part.startswith(".") or part == "__MACOSX" for part in parts):
                continue
            destination = root.joinpath(*parts[1:]).resolve()
            if not destination.is_relative_to(root):
                raise ValueError("Archive path escapes destination")
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite {destination}")
            if entry.file_size > 5_000_000:
                raise ValueError("Unexpectedly large example file")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(entry))
            count += 1
    print(f"Imported {count} files into {root}")


if __name__ == "__main__":
    main()
