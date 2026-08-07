import socket
import argparse


def main():
    parser = argparse.ArgumentParser(description="Connect to inference_stream socket and display JSON events.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17000)
    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((args.host, args.port))
        print(f"Connected to {args.host}:{args.port}")
        data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
            while b"\n" in data:
                line, data = data.split(b"\n", 1)
                if line.strip():
                    print(line.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
