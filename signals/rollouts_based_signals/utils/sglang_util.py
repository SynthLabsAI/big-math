import atexit
import os
import signal
import subprocess
import time

import openai

from .openai_server import OpenAIServerManager


def get_sglang_response(port):
    """
    tries to get a response from the sglang server
    :return:
    """

    client = openai.Client(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")

    # Text completion
    response = client.completions.create(
        model="default",
        prompt="The capital of France is",
        temperature=0,
        max_tokens=1,
    )
    print(response)

def kill_process_group(pid):
    """Kill the entire process group for the given PID."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        pass  # Process already terminated

class SGLangServerManager(OpenAIServerManager):
    def launch_servers(self, model_name, start_port=1234, tp=1, max_time=600):
        """
        Launches an sglang server on all available devices.
        
        Args:
            model_name (str): Path to the model.
            start_port (int): Port to start on.
            tp (int): Tensor parallelism.
            max_time (int): Maximum time (in seconds) to wait for the server to become ready.
        
        Returns:
            tuple: (ports, subprocesses) where ports is a list of ports and subprocesses is a list of Popen objects.
        """
        subprocesses = []
        # Get list of devices from env var (defaulting to 0-7 if not set)
        devices = os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
        dp = len(devices) // tp

        # Correctly generate ports based on tp and available devices
        # (Even if we only launch one process, we keep a list for compatibility.)
        ports = [port for port in range(start_port, start_port + len(devices), tp)]

        # Build the command as a list to avoid using the shell
        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path", model_name,
            "--port", str(start_port),
            "--tp", str(tp),
            "--dp", str(dp),
            "--log-level", "error",
        ]
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")

        # Launch the server process in its own process group.
        process = subprocess.Popen(cmd, start_new_session=True)
        subprocesses.append(process)

        # Ensure that the child process group is killed when the parent exits.
        atexit.register(kill_process_group, process.pid)

        # Optionally, also install signal handlers for SIGINT and SIGTERM.
        def _signal_handler(sig, frame):
            kill_process_group(process.pid)
            raise KeyboardInterrupt

        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        # Wait until at least one port is responsive or timeout is reached.
        start_time = time.monotonic()
        ports_working = []

        while time.monotonic() - start_time < max_time:
            for port in ports:
                if port in ports_working:
                    continue
                try:
                    get_sglang_response(port)
                    ports_working.append(port)
                except (openai.APITimeoutError, openai.APIConnectionError) as err:
                    print(f"Port {port} not ready yet.")
            if ports_working:
                break
            time.sleep(1)  # shorter sleep interval for faster feedback

        else:
            # Timeout reached, ensure cleanup and then raise error.
            kill_process_group(process.pid)
            raise TimeoutError("Server did not become ready within the allotted time.")

        # Restore original signal handlers.
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)

        return ports, subprocesses