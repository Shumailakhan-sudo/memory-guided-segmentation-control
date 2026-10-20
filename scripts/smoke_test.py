                    {
                        "configuration": configuration,
                        "sequence": sequence,
                        "frame_index": frame_index,
                        "jaccard": round(float(rng.uniform(0.8, 1)), 4),
                        "boundary_f": round(float(rng.uniform(0.8, 1)), 4),
                    }
                )

    write_csv_records(directory / "segmentation_frames.csv", segmentation)
    write_csv_records(directory / "configuration_frames.csv", configurations)
    write_csv_records(
        directory / "cartesian_episodes.csv",
        [
            {
                "episode": str(episode),
                "success": 1,
                "confidence": 0.8,
                "episode_s": 3.0,
                "path_m": 0.5,
                "grasp_s": "" if episode == 37 else 2.5,
            }
            for episode in (15, 37, 56)
        ],
    )
    write_csv_records(
        directory / "reward_episodes.csv",
        [
            {
                "episode": str(episode),
                "duration_s": 3.0,
                "cumulative_reward": 6.0,
                "efficiency": 2.0,
            }
            for episode in (15, 45)
        ],
    )


def run(work: Path) -> dict:
    dataset_dir = work / "DAVIS"
    sequences = build_synthetic_davis(dataset_dir)
    records_dir = work / "records"
    build_synthetic_records(records_dir)

    dataset_config = work / "davis.toml"
    dataset_config.write_text(
        (ROOT / "configs" / "davis.toml")
        .read_text(encoding="utf-8")
        .replace('root = "data/raw/DAVIS"', f'root = "{dataset_dir.as_posix()}"'),
        encoding="utf-8",
    )

    simulation_config = work / "sim.toml"
    simulation_config.write_text(
        (ROOT / "configs" / "cartesian_simulation.toml")
        .read_text(encoding="utf-8")
        .replace(
            '"data/original/cartesian_episodes.csv"',
            f'"{(records_dir / "cartesian_episodes.csv").as_posix()}"',
        )
        .replace(
            '"data/original/reward_episodes.csv"',
            f'"{(records_dir / "reward_episodes.csv").as_posix()}"',
        )
        .replace(
            'dir = "results/simulation"',
            f'dir = "{(work / "out" / "simulation").as_posix()}"',
        ),
        encoding="utf-8",
    )

    output_dir = work / "out"
    exit_codes = {
        "prepare_davis": prepare_davis.main(
            [
                "--config", str(dataset_config),
                "--manifest", str(output_dir / "manifest.csv"),
                "--data-kind", "synthetic",
            ]
        ),
        "visibility": run_visibility_analysis.main(
            [
                "--config", str(dataset_config),
                "--results-root", str(output_dir),
                "--data-kind", "synthetic",
                "--max-tau", "5",
            ]
        ),
        "calibration": run_calibration.main(
            [
                "--records", str(records_dir / "calibration_pairs.csv"),
                "--data-kind", "synthetic",
                "--out", str(output_dir / "calibration"),
            ]
        ),
        "configurations": run_configuration_comparison.main(
            [
                "--records", str(records_dir / "configuration_frames.csv"),
                "--data-kind", "synthetic",
                "--out", str(output_dir / "configurations"),
            ]
        ),
        "simulation": run_cartesian_simulation.main(
            [
                "--config", str(simulation_config),
                "--summarise",
                "--data-kind", "synthetic",
            ]
        ),
    }
    return {"sequences": sequences, "exit_codes": exit_codes, "out": output_dir}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run integration checks with synthetic data.")
    parser.add_argument("--keep", help="Store generated files in the specified directory.")
    args = parser.parse_args(argv)

    work = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="mgvos_smoke_"))
    try:
        result = run(work)
        print("Synthetic test exit codes:", result["exit_codes"])
        return 0 if all(code == 0 for code in result["exit_codes"].values()) else 1
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
