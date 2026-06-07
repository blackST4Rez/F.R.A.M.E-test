"""Clean image files in the train and test dataset folders.

This script scans the `model/new_data/train` and `model/new_data/test`
directories and removes any files that are not valid image files.
It also removes zero-byte files and optionally empty directories.
"""

import argparse
import hashlib
import os

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}


def hash_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def detect_image_type(path: str) -> str | None:
    with open(path, 'rb') as handle:
        header = handle.read(16)

    if header.startswith(b'\xFF\xD8\xFF'):
        return 'jpeg'
    if header.startswith(b'\x89PNG\r\n\x1A\n'):
        return 'png'
    if header[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'
    if header.startswith(b'BM'):
        return 'bmp'
    if header.startswith((b'II*\x00', b'MM\x00*')):
        return 'tiff'
    if header.startswith(b'RIFF') and header[8:12] == b'WEBP':
        return 'webp'
    return None


def is_image_file(path: str) -> bool:
    if not os.path.isfile(path):
        return False

    if os.path.getsize(path) == 0:
        return False

    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False

    return detect_image_type(path) is not None


def clean_folder(root_dir: str, dry_run: bool = False, remove_empty_dirs: bool = True, remove_duplicates: bool = False) -> dict:
    stats = {
        'checked': 0,
        'removed': 0,
        'skipped': 0,
        'duplicates': 0,
        'dirs_removed': 0,
        'persons': {},
    }
    seen_hashes: dict[str, str] = {}

    def person_for_path(path: str) -> str:
        rel = os.path.relpath(path, root_dir)
        return rel.split(os.sep)[0] if rel and rel != os.curdir else ''

    for base, dirs, files in os.walk(root_dir, topdown=False):
        person = person_for_path(base)
        for filename in files:
            filepath = os.path.join(base, filename)
            stats['checked'] += 1
            if person not in stats['persons']:
                stats['persons'][person] = {
                    'checked': 0,
                    'removed': 0,
                    'kept': 0,
                    'duplicates': 0,
                }
            stats['persons'][person]['checked'] += 1

            if not is_image_file(filepath):
                stats['removed'] += 1
                stats['persons'][person]['removed'] += 1
                if dry_run:
                    print(f"[DRY RUN] Remove file: {filepath}")
                else:
                    print(f"Removing file: {filepath}")
                    os.remove(filepath)
                continue

            if remove_duplicates:
                file_hash = hash_file(filepath)
                if file_hash in seen_hashes:
                    stats['removed'] += 1
                    stats['duplicates'] += 1
                    stats['persons'][person]['removed'] += 1
                    stats['persons'][person]['duplicates'] += 1
                    if dry_run:
                        print(f"[DRY RUN] Duplicate remove: {filepath} (duplicate of {seen_hashes[file_hash]})")
                    else:
                        print(f"Removing duplicate file: {filepath} (duplicate of {seen_hashes[file_hash]})")
                        os.remove(filepath)
                    continue
                seen_hashes[file_hash] = filepath

            stats['skipped'] += 1
            stats['persons'][person]['kept'] += 1

        if remove_empty_dirs and not dry_run and not os.listdir(base):
            os.rmdir(base)
            stats['dirs_removed'] += 1
            print(f"Removed empty directory: {base}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description='Clean train/test image folders.')
    parser.add_argument(
        '--data-dir',
        default='model/new_data',
        help='Base data directory containing train/ and test/ subfolders.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be removed without deleting files.',
    )
    parser.add_argument(
        '--no-rm-empty-dirs',
        action='store_true',
        help='Do not remove empty directories after cleanup.',
    )
    parser.add_argument(
        '--remove-duplicates',
        action='store_true',
        help='Remove duplicate image files within each subset.',
    )

    args = parser.parse_args()
    base_dir = os.path.abspath(args.data_dir)

    if not os.path.isdir(base_dir):
        raise SystemExit(f"Data directory not found: {base_dir}")

    for subset in ['train', 'test']:
        subset_dir = os.path.join(base_dir, subset)
        if not os.path.isdir(subset_dir):
            print(f"Skipping missing subset: {subset_dir}")
            continue

        print(f"Cleaning subset: {subset_dir}")
        stats = clean_folder(
            subset_dir,
            dry_run=args.dry_run,
            remove_empty_dirs=not args.no_rm_empty_dirs,
        )
        print(
            f"Summary for {subset}: checked={stats['checked']}, "
            f"removed={stats['removed']}, skipped={stats['skipped']}, "
            f"duplicates={stats['duplicates']}, dirs_removed={stats['dirs_removed']}"
        )
        if stats['persons']:
            print('Per-person cleaned image counts:')
            for person, person_stats in sorted(stats['persons'].items()):
                if not person:
                    continue
                print(
                    f"  {person}: checked={person_stats['checked']}, "
                    f"removed={person_stats['removed']}, "
                    f"duplicates={person_stats['duplicates']}, "
                    f"cleaned={person_stats['kept']}"
                )
        print()

    if args.no_rm_empty_dirs:
        print('Note: empty directory removal disabled (--no-rm-empty-dirs).')


if __name__ == '__main__':
    main()
