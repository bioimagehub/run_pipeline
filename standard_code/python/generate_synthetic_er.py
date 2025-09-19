import os
import sys
import argparse


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic ER images using ERnet-v2's Simulate_ER script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-folder", required=True, help="Folder to write synthetic images (will be created)")
    parser.add_argument("--N", type=int, default=6, help="Number of images to generate")
    parser.add_argument("--radius", type=int, default=250, help="Canvas radius/pixels used by generator")
    parser.add_argument("--dpi", type=int, default=64, help="Figure DPI used by generator")
    parser.add_argument("--break-prob", type=float, default=0.0, help="Probability to break edges (0..1)")
    parser.add_argument("--num-seeds", type=int, default=400, help="Number of seeds for Voronoi generation")
    parser.add_argument("--sigma", type=float, default=2.0, help="Smoothing sigma inside generator")
    parser.add_argument("--darkness", action="store_true", help="Enable darkness modulation in generator")
    parser.add_argument(
        "--ernet-sim-path",
        default=None,
        help="Optional explicit path to ERnet-v2/Training/Simulate_ER folder. Defaults to repo external path.",
    )
    args = parser.parse_args(argv)

    # Resolve path to ERnet-v2 Simulate_ER
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    if args.ernet_sim_path:
        sim_dir = os.path.abspath(args.ernet_sim_path)
    else:
        sim_dir = os.path.join(repo_root, "external", "ERnet-v2", "Training", "Simulate_ER")

    if not os.path.isdir(sim_dir):
        print(f"Simulate_ER folder not found: {sim_dir}")
        return 2

    # Ensure we can import the generator and its local dependency delaunay2D.py
    if sim_dir not in sys.path:
        sys.path.insert(0, sim_dir)

    # The upstream script imports `parser` (removed from stdlib in Py3.11). Provide a dummy.
    try:
        import types as _types
        if 'parser' not in sys.modules:
            sys.modules['parser'] = _types.ModuleType('parser')  # type: ignore
    except Exception:
        pass

    try:
        from Simulate_ER_images_script import generate_ER_images  # type: ignore
    except Exception as e:
        print(f"Failed importing generate_ER_images from {sim_dir}: {e}")
        return 3

    out_dir = os.path.abspath(args.output_folder)
    os.makedirs(out_dir, exist_ok=True)

    # The upstream generator writes a subfolder named by 'foldername'.
    # We force it to be exactly the provided output folder for simplicity.
    generate_ER_images(
        radius=int(args.radius),
        dpi=int(args.dpi),
        breakEdgesProbability=float(args.break_prob),
        N=int(args.N),
        numSeeds=int(args.num_seeds),
        foldername=out_dir,
        sigma=float(args.sigma),
        darkness=bool(args.darkness),
    )

    # Provide a tiny hint on produced files
    print(f"Synthetic ER images written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
