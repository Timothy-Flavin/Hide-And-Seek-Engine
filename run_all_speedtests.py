import os
import re
import shutil
import subprocess
import sys


def update_gravity_h(num_agents: int, filepath: str = "src/gravity.h"):
    with open(filepath, "r") as f:
        content = f.read()

    # Replace the N_AGENTS constexpr definition
    new_content = re.sub(
        r"constexpr\s+int\s+N_AGENTS\s*=\s*\d+\s*;",
        f"constexpr int N_AGENTS = {num_agents};",
        content,
    )

    with open(filepath, "w") as f:
        f.write(new_content)

    print(f"Updated {filepath} to use {num_agents} agents.")


def clean_build_directory():
    build_dir = "build"
    if os.path.exists(build_dir):
        print(f"Removing '{build_dir}' directory to force recompilation...")
        shutil.rmtree(build_dir, ignore_errors=True)


def recompile():
    print("Recompiling the engine...")
    # Using uv pip install . to build the package
    result = subprocess.run(["uv", "pip", "install", "."], capture_output=False)
    if result.returncode != 0:
        print("Compilation failed!")
        sys.exit(1)


def run_speedtest(num_agents: int):
    print(f"Running speedtest for {num_agents} agents...")
    result = subprocess.run(
        [sys.executable, "speedtest_old.py", "--agents", str(num_agents)],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"Speedtest failed for {num_agents} agents!")
        sys.exit(1)


def main():
    agent_counts = list(range(1, 11))

    for n in agent_counts:
        print(f"\n{'='*50}")
        print(f"Starting test suite for N_AGENTS = {n}")
        print(f"{'='*50}")

        update_gravity_h(n)
        clean_build_directory()
        recompile()
        run_speedtest(n)

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()
