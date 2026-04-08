import subprocess
import sys
import time


def main():
    modes = ["centralized", "decentralized", "no_obs"]
    requires_states = ["True", "False"]

    for mode in modes:
        for state in requires_states:
            print(f"\n==============================================")
            print(f"Running speedtest for mode={mode} state={state}")
            print(f"==============================================\n")

            cmd = [
                sys.executable,
                "speedtest.py",
                "--mode",
                mode,
                "--requires_state",
                state,
            ]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"Failed on {mode} state={state}")
                sys.exit(1)
            time.sleep(60)  # let the cpu cool down
    print("\nAll speedtests completed successfully.")


if __name__ == "__main__":
    main()
