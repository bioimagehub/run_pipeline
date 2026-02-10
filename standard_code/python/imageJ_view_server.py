import os
import sys
import subprocess
import socket
import pickle
import struct
import numpy as np
import signal
# Initialize ImageJ in interactive mode 
# You can start this server in a separate terminal and then run the client code to send images to it 
# uv run --group imagej python .\standard_code\python\imageJ_view_server.py


REQUIRED_JAVA_MAJOR = 11
FIJI_PATH = r"E:\Oyvind\OF_program_files\OF_Fiji.app"
HOST = "127.0.0.1"
PORT = 50007
SERVER_VERSION = "2026-02-10a"

shutdown_requested = False
ij_instance = None
JAVA_HOME_OVERRIDE_ENV = "IMAGEJ_JAVA_HOME"


def _handle_sigint(signum, frame):
    global shutdown_requested
    print("\n[INFO] Ctrl+C received. Shutting down server...")
    shutdown_requested = True
    if ij_instance is not None:
        try:
            ij_instance.ui().dispose()
        except Exception:
            pass


signal.signal(signal.SIGINT, _handle_sigint)


def _check_java():
    java_cmd = "java"
    java_home_override = os.environ.get(JAVA_HOME_OVERRIDE_ENV)
    if java_home_override:
        os.environ["JAVA_HOME"] = java_home_override
        candidate = os.path.join(java_home_override, "bin", "java.exe")
        if os.path.exists(candidate):
            java_cmd = candidate
        else:
            print(
                f"[WARN] {JAVA_HOME_OVERRIDE_ENV} set, but java.exe not found. "
                "Falling back to PATH."
            )
    try:
        result = subprocess.run(
            [java_cmd, "-version"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("Java not found on PATH.")
        sys.exit(1)

    version_output = result.stderr or result.stdout
    import re
    match = re.search(r'version "([^"]+)"', version_output)
    if not match:
        print("Could not parse Java version.")
        sys.exit(1)

    version_str = match.group(1)

    if version_str.startswith("1."):
        major = int(version_str.split(".")[1])
    else:
        major = int(version_str.split(".")[0])

    if major < REQUIRED_JAVA_MAJOR:
        print(f"Java {version_str} detected. Java 11+ required.")
        sys.exit(1)

    print(f"[OK] Java {version_str} detected.")
    if major != 11:
        print(
            "[WARN] Java 11 is recommended to avoid the JavaScript engine popup. "
            f"Set {JAVA_HOME_OVERRIDE_ENV} to your Java 11 install path."
        )


def _init_imagej():
    try:
        import scyjava
        import imagej
        print("Initializing ImageJ...")
        scyjava.config.add_option(
            "-Dscijava.plugin.blacklist="
            "org.scijava.plugins.scripting.javascript.JavaScriptScriptLanguage,"
            "org.scijava.plugins.scripting.javascript.JavaScriptLanguage,"
            "org.scijava.plugins.scripting.jsr223.JSR223ScriptLanguage,"
            "org.scijava.plugins.scripting.jsr223.ScriptEngineLanguage"
        )
        scyjava.config.add_option(
            "-Dij1.plugin.blacklist=ij.plugin.JavaScript,ij.plugin.JavaScript_"
        )
        scyjava.config.add_option("-Dscijava.scripting.languages=")
        scyjava.config.add_option("-Dscijava.scripting.enabled=false")
        ij = imagej.init(
            FIJI_PATH,
            mode="interactive",
            add_legacy=True,
        )
        print("[OK] ImageJ initialized.")
        return ij
    except Exception as e:
        print("\nImageJ failed to initialize:\n")
        print(e)
        sys.exit(1)


def is_server_running(host=HOST, port=PORT, timeout=1.0, verbose=False):
    """Return True if the ImageJ server socket accepts a connection."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as conn:
            conn.sendall(struct.pack(">Q", 0))
            if verbose:
                print(f"[OK] ImageJ server found at {host}:{port}")
            return True
    except OSError:
        if verbose:
            print(f"[WARN] ImageJ server not reachable at {host}:{port}")
        return False


def show_image(arr, host=HOST, port=PORT, timeout=5.0, verbose=False):
    """Send a numpy array to a running ImageJ server via socket."""
    if not is_server_running(host=host, port=port, timeout=timeout, verbose=verbose):
        raise RuntimeError(
            f"ImageJ server is not running at {host}:{port}."
        )
    data = pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        with socket.create_connection((host, port), timeout=timeout) as conn:
            if verbose:
                print(f"[INFO] Sending {len(data)} bytes to {host}:{port}")
            conn.sendall(struct.pack(">Q", len(data)))
            if verbose:
                print("[INFO] Sent header")
            conn.sendall(data)
            if verbose:
                print("[INFO] Sent payload")
            try:
                conn.shutdown(socket.SHUT_WR)
            except OSError:
                pass
    except OSError as e:
        raise RuntimeError(
            f"Failed to send image to ImageJ server at {host}:{port}."
        ) from e

def main():
    _check_java()
    global ij_instance
    ij = _init_imagej()
    ij_instance = ij
    ij.ui().showUI()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()

        print(f"[INFO] Server version: {SERVER_VERSION} (pid={os.getpid()})")
        print(f"ImageJ server running on {HOST}:{PORT}")
        print("Close ImageJ window to stop.")

        try:
            while not shutdown_requested:
                s.settimeout(1.0)  # allows periodic shutdown check
                try:
                    conn, addr = s.accept()
                except socket.timeout:
                    continue

                with conn:
                    conn.settimeout(120.0)
                    print(f"[INFO] Connection from {addr[0]}:{addr[1]}")
                    header = b""
                    while len(header) < 8:
                        try:
                            chunk = conn.recv(8 - len(header))
                        except socket.timeout:
                            print(
                                f"[WARN] Header read timeout (got {len(header)} bytes)"
                            )
                            break
                        if not chunk:
                            break
                        header += chunk

                    if len(header) < 8:
                        print("[INFO] Empty payload received; ignoring.")
                        continue

                    expected = struct.unpack(">Q", header)[0]
                    if expected == 0:
                        print("[INFO] Probe received; no payload.")
                        continue
                    print(f"[INFO] Expecting {expected} bytes")
                    data = bytearray()
                    next_report = 50 * 1024 * 1024
                    while len(data) < expected:
                        try:
                            packet = conn.recv(min(256 * 1024, expected - len(data)))
                        except socket.timeout:
                            print(
                                f"[WARN] Payload read timeout at {len(data)} of {expected} bytes"
                            )
                            break
                        if not packet:
                            break
                        data.extend(packet)
                        if len(data) >= next_report:
                            print(f"[INFO] Received {len(data)} bytes...")
                            next_report += 50 * 1024 * 1024

                    if len(data) != expected:
                        print(
                            f"[WARN] Incomplete payload: got {len(data)} of {expected} bytes"
                        )
                        continue

                    print(f"[INFO] Payload size: {len(data)} bytes")

                    try:
                        arr = np.asarray(pickle.loads(data))
                    except Exception as e:
                        print("[WARN] Failed to decode payload; ignoring.")
                        print(e)
                        continue
                    axes = None
                    if arr.ndim == 5:
                        axes = "TCZYX"
                    elif arr.ndim == 4:
                        axes = "CZYX"
                    elif arr.ndim == 3:
                        axes = "ZYX"

                    print(
                        f"[INFO] Received array shape={arr.shape}, dtype={arr.dtype}"
                    )
                    try:
                        if axes:
                            try:
                                dataset = ij.py.to_dataset(arr, axes=axes)
                            except TypeError:
                                print(
                                    "[WARN] to_dataset() does not support axes; using default ordering"
                                )
                                dataset = ij.py.to_dataset(arr)
                        else:
                            dataset = ij.py.to_dataset(arr)
                        ij.ui().show(dataset)
                        print("[OK] Image displayed in ImageJ")
                    except Exception as e:
                        print("[ERROR] Failed to display image in ImageJ")
                        print(e)

        except Exception as e:
            print("\n[ERROR] Server exception:")
            print(e)

        finally:
            print("[INFO] Closing ImageJ...")
            # try:
            #     ij.dispose()
            # except Exception:
            #     pass

            print("[INFO] Server stopped cleanly.")
            sys.exit(0)


if __name__ == "__main__":
    main()
